"""Evaluate a trained PPO checkpoint on ``swm/PushT-v1`` and optionally record video.

    python -m src.ppo.evaluate --checkpoint runs/ppo_pusht__seed1/final.pt \
        --episodes 20 --video
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.ppo.envs import make_env
from src.ppo.networks import ActorCritic
from src.ppo.utils import ObsNormalizer, get_device


def load_agent(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["act_dim"]
    hidden = tuple(ckpt["config"].get("hidden_dims", (256, 256)))
    agent = ActorCritic(obs_dim, act_dim, hidden).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    obs_norm = None
    if "obs_norm" in ckpt:
        obs_norm = ObsNormalizer((obs_dim,), clip=ckpt["config"].get("obs_clip", 10.0))
        obs_norm.load_state_dict(ckpt["obs_norm"])
    return agent, obs_norm, ckpt["config"]


@torch.no_grad()
def evaluate(
    checkpoint: str,
    episodes: int = 20,
    deterministic: bool = True,
    record_video: bool = False,
    video_dir: str = "runs/eval_videos",
    seed: int = 0,
    device: str = "auto",
) -> dict:
    dev = get_device(device)
    agent, obs_norm, cfg = load_agent(checkpoint, dev)

    env = make_env(
        cfg.get("env_id", "swm/PushT-v1"),
        seed=seed,
        max_episode_steps=cfg.get("max_episode_steps", 200),
        record_stats=True,
    )()

    returns, lengths, successes, final_dists = [], [], [], []
    frames_all = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        frames = []
        info = {}
        while not done:
            if record_video:
                frames.append(env.render())
            norm = obs_norm.normalize(obs[None], update=False) if obs_norm else obs[None]
            x = torch.as_tensor(norm, dtype=torch.float32, device=dev)
            mean = agent.actor_mean(x)
            action = mean if deterministic else agent.get_action_and_value(x)[0]
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

        returns.append(float(info["episode"]["r"]))
        lengths.append(float(info["episode"]["l"]))
        successes.append(float(info.get("is_success", False)))
        final_dists.append(float(-reward))  # native reward is -distance
        if record_video and frames:
            frames_all.append((ep, frames, bool(info.get("is_success", False))))

    env.close()

    if record_video and frames_all:
        _write_videos(frames_all, video_dir)

    summary = {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_final_distance": float(np.mean(final_dists)),
    }
    print("Evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def _write_videos(frames_all, video_dir: str) -> None:
    import imageio

    out = Path(video_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ep, frames, success in frames_all:
        tag = "success" if success else "fail"
        path = out / f"episode_{ep:03d}_{tag}.mp4"
        imageio.mimsave(path, [np.asarray(f) for f in frames], fps=10)
        print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO on swm/PushT-v1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    parser.add_argument("--video", action="store_true", help="record mp4 videos")
    parser.add_argument("--video-dir", type=str, default="runs/eval_videos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        record_video=args.video,
        video_dir=args.video_dir,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
