"""Evaluate a trained *latent* PPO checkpoint on ``swm/PushT-v1`` (+ optional video).

Counterpart to :mod:`src.ppo.evaluate` for the frozen-encoder latent agent
(:mod:`src.ppo.latent_agent`). One policy decision is an open-loop chunk of
``action_chunk_size`` env steps; the policy input is the dilated stack of
per-frame LeWM latents (:class:`src.ppo.latent_env.LatentHistory`), exactly as
during training. Env settings (``fixed_target``, ``max_episode_steps``, ...) and
the agent contract are read from the checkpoint so evaluation matches training::

    python -m src.ppo.evaluate_latent \
        --checkpoint runs/latent_ppo_pusht__seed1/08072026-133525/best.pt \
        --episodes 20 --video
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.envs import PUSHT_FIXED_TARGET_POSE
from src.ppo.latent_agent import build_latent_agent
from src.ppo.latent_env import LatentHistory, make_latent_env, success_from_info
from src.ppo.lewm_encoder import LeWMLatentEncoder
from src.ppo.utils import get_device

_ENCODER_PREFIXES = ("actor.encoder.", "critic.encoder.")


def load_latent_agent(checkpoint: str, device: torch.device):
    """Rebuild the latent PPO agent (frozen encoder + trained policy/value)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    contract = ckpt.get("contract", cfg)

    encoder = LeWMLatentEncoder.from_checkpoint(
        device,
        cfg.get("encoder_checkpoint"),
        latent_dim=int(contract["latent_dim"]),
    )
    agent = build_latent_agent(
        encoder=encoder,
        latent_dim=int(contract["latent_dim"]),
        frame_stack=int(contract["frame_stack"]),
        action_dim=int(contract["action_dim"]),
        action_chunk_size=int(contract["action_chunk_size"]),
        hidden_dim=int(contract["hidden_dim"]),
        init_log_std=float(cfg.get("init_log_std", 0.0)),
        bc_checkpoint_path=None,  # weights come from the PPO checkpoint below
        device=device,
    )
    # The saved state omits the (frozen, large) encoder weights, which the loader
    # above already restored -- so the only allowed missing keys are encoder keys.
    missing, unexpected = agent.load_state_dict(ckpt["agent"], strict=False)
    stray = [k for k in missing if not k.startswith(_ENCODER_PREFIXES)]
    if stray or unexpected:
        raise RuntimeError(f"checkpoint mismatch (missing={stray}, unexpected={unexpected})")
    agent.eval()
    return agent, encoder, cfg, contract


@torch.no_grad()
def _encode(encoder, obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """``[H, W, C]`` uint8 frame -> ``[latent_dim]`` latent."""
    t = torch.as_tensor(np.asarray(obs), device=device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t.permute(0, 3, 1, 2).contiguous()
    return encoder(t)[0]


@torch.no_grad()
def _predict_chunk(agent, stacked: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """``[1, F, D]`` latents -> ``[k, adim]`` clamped action chunk (on CPU)."""
    if deterministic:
        action = agent.actor.dist_from_latents(stacked).mean
    else:
        action, _, _, _ = agent.get_action_and_value_from_latents(stacked)
    return torch.clamp(action, -1.0, 1.0)[0].cpu()


def _temporal_ensemble(predictions: list[torch.Tensor], decay: float) -> torch.Tensor:
    """ACT-style weighted average of overlapping predictions for one timestep.

    Mirrors ``src.bc.run_eval.temporal_ensemble_action``: newer predictions get
    exponentially more weight (``decay``); ``decay=0`` is a plain average.
    """
    stacked = torch.stack(predictions, dim=0)  # [n, adim]
    if decay == 0.0 or len(predictions) == 1:
        return stacked.mean(dim=0)
    age = torch.arange(len(predictions), dtype=stacked.dtype)
    weights = torch.exp(-decay * age)
    weights = weights / weights.sum()
    return (stacked * weights[:, None]).sum(dim=0)


@torch.no_grad()
def _run_episode(
    env,
    agent,
    encoder,
    contract: dict,
    dev: torch.device,
    *,
    seed: int,
    deterministic: bool,
    replan_interval: int,
    temporal_ensemble: bool,
    te_decay: float,
    record_video: bool,
) -> dict:
    """Roll out one episode under the chosen replanning strategy.

    * ``temporal_ensemble``: query every step, average overlapping chunk votes
      (implies the deterministic mean action).
    * otherwise receding horizon: predict a chunk, execute ``replan_interval`` of
      its ``k`` actions open-loop, then re-plan (``replan_interval >= k`` is pure
      open-loop; ``1`` re-plans every step).
    """
    k = int(contract["action_chunk_size"])
    obs, info = env.reset(seed=seed)
    hist = LatentHistory(int(contract["frame_stack"]), int(contract["frame_stride"]))
    hist.append(_encode(encoder, obs, dev))
    frames = [obs] if record_video else None
    done = terminated = False
    last_reward = 0.0

    if temporal_ensemble:
        from collections import defaultdict

        buffers: dict[int, list[torch.Tensor]] = defaultdict(list)
        t = 0
        while not done:
            chunk = _predict_chunk(agent, hist.stacked().unsqueeze(0), True)
            for offset in range(k):
                buffers[t + offset].append(chunk[offset])
            action = _temporal_ensemble(buffers.pop(t), te_decay)
            obs, reward, terminated, truncated, info = env.step(
                torch.clamp(action, -1.0, 1.0).numpy()
            )
            hist.append(_encode(encoder, obs, dev))
            last_reward = float(reward)
            if record_video:
                frames.append(obs)
            t += 1
            done = bool(terminated or truncated)
    else:
        stride = replan_interval if replan_interval > 0 else k
        while not done:
            chunk = _predict_chunk(agent, hist.stacked().unsqueeze(0), deterministic).numpy()
            for j in range(min(stride, k)):
                obs, reward, terminated, truncated, info = env.step(chunk[j])
                hist.append(_encode(encoder, obs, dev))
                last_reward = float(reward)
                if record_video:
                    frames.append(obs)
                if terminated or truncated:
                    done = True
                    break

    return {
        "return": float(info["episode"]["r"]),
        "length": float(info["episode"]["l"]),
        "success": success_from_info(info, terminated),
        "final_dist": float(-last_reward),  # native reward is -distance
        "frames": frames,
    }


@torch.no_grad()
def run_evaluation(
    *,
    agent,
    encoder,
    contract: dict,
    dev: torch.device,
    episodes: int,
    deterministic: bool,
    record_video: bool,
    video_dir: str,
    video_resolution: int,
    replan_interval: int,
    temporal_ensemble: bool,
    temporal_ensemble_decay: float,
    env_id: str,
    max_episode_steps: int,
    fixed_target: bool,
    fixed_target_pose,
    fixed_target_block_success: bool,
    seed: int,
    block_start_near_goal: bool = False,
    block_start_radius: float = 50.0,
) -> dict:
    """Roll out ``episodes`` under a resolved env + strategy config and summarise.

    The harness shared by :func:`evaluate` (PPO checkpoints) and
    :func:`src.ppo.evaluate_bc_latent.evaluate` (BC checkpoints): identical env,
    dilated ``LatentHistory``, replanning strategy, success metric, and summary --
    so BC and PPO numbers are directly comparable. All config is already resolved
    by the caller (no checkpoint defaults are consulted here).
    """
    k = int(contract["action_chunk_size"])
    pose = np.asarray(fixed_target_pose, dtype=float)
    if fixed_target:
        print(f"Goal: FIXED at pose {np.round(pose, 3).tolist()} (block-success={fixed_target_block_success})")
    else:
        print("Goal: RE-RANDOMIZED every episode (native env, fixed_target off)")
    if block_start_near_goal:
        print(f"Block start: within {block_start_radius:g}px of the green T center")

    if temporal_ensemble:
        mode = f"temporal-ensemble (query every step, decay={temporal_ensemble_decay})"
    elif replan_interval and replan_interval < k:
        mode = f"receding-horizon (re-plan every {replan_interval} step(s), chunk={k})"
    else:
        mode = f"open-loop (execute full chunk={k})"
    print(f"Strategy: {mode} | action={'mean' if deterministic else 'sampled'}")

    env = make_latent_env(
        env_id=env_id,
        seed=seed,
        idx=0,
        max_episode_steps=max_episode_steps,
        record_stats=True,
        fixed_target=fixed_target,
        fixed_target_pose=pose,
        fixed_target_block_success=fixed_target_block_success,
        block_start_near_goal=block_start_near_goal,
        block_start_radius=block_start_radius,
    )()

    returns, lengths, successes, final_dists = [], [], [], []
    frames_all = []

    for ep in range(episodes):
        result = _run_episode(
            env,
            agent,
            encoder,
            contract,
            dev,
            seed=seed + ep,
            deterministic=deterministic,
            replan_interval=replan_interval,
            temporal_ensemble=temporal_ensemble,
            te_decay=temporal_ensemble_decay,
            record_video=record_video,
        )
        returns.append(result["return"])
        lengths.append(result["length"])
        successes.append(result["success"])
        final_dists.append(result["final_dist"])
        print(
            f"  episode {ep:03d}: success={result['success']:.0f} return={result['return']:8.1f} "
            f"len={result['length']:5.0f} final_dist={result['final_dist']:6.1f}"
        )
        if record_video:
            frames_all.append((ep, result["frames"], bool(result["success"])))

    env.close()

    if record_video and frames_all:
        _write_videos(frames_all, video_dir, video_resolution)

    summary = {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_final_distance": float(np.mean(final_dists)),
    }
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


@torch.no_grad()
def evaluate(
    checkpoint: str,
    episodes: int = 20,
    deterministic: bool = True,
    record_video: bool = False,
    video_dir: str | None = None,
    video_resolution: int = 512,
    replan_interval: int = 0,
    temporal_ensemble: bool = False,
    temporal_ensemble_decay: float = 0.01,
    fixed_target: bool | None = None,
    fixed_target_pose=None,
    fixed_target_block_success: bool | None = None,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    dev = get_device(device)
    agent, encoder, cfg, contract = load_latent_agent(checkpoint, dev)

    # Goal configuration: default to what the checkpoint trained with; CLI may
    # override to test other/changing goals. ``fixed_target=False`` re-randomizes
    # the goal T every episode (native env); a custom ``fixed_target_pose`` keeps
    # one fixed goal but relocates it.
    ft = cfg.get("fixed_target", False) if fixed_target is None else fixed_target
    ftbs = (
        cfg.get("fixed_target_block_success", True)
        if fixed_target_block_success is None
        else fixed_target_block_success
    )
    pose = PUSHT_FIXED_TARGET_POSE if fixed_target_pose is None else np.asarray(fixed_target_pose, dtype=float)
    # Match the block start distribution the checkpoint trained with.
    block_start_near_goal = bool(cfg.get("block_start_near_goal", False))
    block_start_radius = float(cfg.get("block_start_radius", 50.0))

    # Default: a folder named after the checkpoint, beside it (e.g. the "best"
    # checkpoint's videos land in ``<run>/best/``).
    if video_dir is None:
        video_dir = str(Path(checkpoint).with_suffix(""))

    return run_evaluation(
        agent=agent,
        encoder=encoder,
        contract=contract,
        dev=dev,
        episodes=episodes,
        deterministic=deterministic,
        record_video=record_video,
        video_dir=video_dir,
        video_resolution=video_resolution,
        replan_interval=replan_interval,
        temporal_ensemble=temporal_ensemble,
        temporal_ensemble_decay=temporal_ensemble_decay,
        env_id=cfg.get("env_id", "swm/PushT-v1"),
        max_episode_steps=cfg.get("max_episode_steps", 300),
        fixed_target=ft,
        fixed_target_pose=pose,
        fixed_target_block_success=ftbs,
        block_start_near_goal=block_start_near_goal,
        block_start_radius= block_start_radius,
        seed=seed,
    )


def _upscale(frame: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a ``[H, W, 3]`` frame to ``resolution x resolution``.

    The native PushT render is only 96x96 (the encoder resizes to 224 anyway, so
    the policy sees the same input regardless). We upscale the recorded frames
    for a watchable video; nearest-neighbour keeps the shapes crisp instead of
    inventing detail.
    """
    frame = np.asarray(frame)
    if resolution <= 0 or frame.shape[:2] == (resolution, resolution):
        return frame
    import cv2

    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)


def _write_videos(frames_all, video_dir: str, resolution: int = 512) -> None:
    import imageio

    out = Path(video_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ep, frames, success in frames_all:
        tag = "success" if success else "fail"
        path = out / f"episode_{ep:03d}_{tag}.mp4"
        imageio.mimsave(path, [_upscale(f, resolution) for f in frames], fps=10)
        print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent PPO on swm/PushT-v1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of the mean"
    )
    parser.add_argument("--video", action="store_true", help="record mp4 videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="output dir for videos (default: a folder named after the checkpoint, beside it)",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        default=512,
        help="square pixel size of saved videos (the 96px render is upscaled; 0 keeps native)",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=0,
        help="receding horizon: re-plan after this many chunk steps (0/>=k = open-loop, 1 = every step)",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="ACT-style: query every step and average overlapping chunk predictions (uses the mean action)",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.01,
        help="exponential decay for temporal-ensemble weights (0.0 = uniform average)",
    )
    # ---- goal configuration (defaults to the checkpoint's training setting) ----
    parser.add_argument(
        "--fixed-target",
        dest="fixed_target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="--fixed-target keeps one fixed goal; --no-fixed-target re-randomizes the goal each episode",
    )
    parser.add_argument(
        "--fixed-target-pose",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "ANGLE"),
        help="relocate the fixed goal T (default: 256 256 0.785 = center, 45 deg)",
    )
    parser.add_argument(
        "--fixed-target-block-success",
        dest="fixed_target_block_success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use block-pose success/reward (vs full-state) in fixed-target mode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        record_video=args.video,
        video_dir=args.video_dir,
        video_resolution=args.video_resolution,
        replan_interval=args.replan_interval,
        temporal_ensemble=args.temporal_ensemble,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        fixed_target=args.fixed_target,
        fixed_target_pose=args.fixed_target_pose,
        fixed_target_block_success=args.fixed_target_block_success,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
