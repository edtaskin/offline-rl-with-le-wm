"""RQ2, part 3: look at the frames PPO believed were successes.

Rolls a *trained* dream-PPO policy inside the same world model it was trained in,
catches every episode the frozen probe terminates as a success, and decodes the
imagined latent it made that call on. The result is a contact sheet of the
agent's own claimed successes -- the direct visual test for whether the policy
found states that satisfy the probe without satisfying the task.

With ``--verify-in-sim`` the eyeball test becomes a measurement. Every dream
episode is seeded from a known expert-dataset frame, so the exact same action
chunks can be replayed in the real simulator from that exact state, and the
probe's verdict compared against the simulator's. That turns "how many of these
look wrong" into a hallucination rate.

Two caveats worth stating on the poster:

* the replay is open-loop -- it executes the actions imagination produced, not
  the actions the policy would choose given real observations, which is the
  right comparison for "was the imagined trajectory real", not for "how good is
  the policy" (that is RQ1's job);
* a mismatch can come from the probe misreading a correct latent or from the
  world model drifting. ``scripts/rq2/probe_horizon.py`` separates those two.

Usage::

    python -m scripts.rq2.dream_success_gallery --seed 1
    python -m scripts.rq2.dream_success_gallery --seed 1 --verify-in-sim --episodes 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.rq_common import find_run_dir, repo_path

STATUS_COLORS = {"good": "#0ca30c", "critical": "#d03b3b", "unknown": "#898781"}
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
TILE = 224


def load_font(size: int):
    """A real TTF at a chosen size.

    PIL's built-in bitmap font is fixed at ~11px and has no glyphs outside
    Latin-1, which renders check marks as tofu; matplotlib always ships
    DejaVu Sans, so borrow that.
    """
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001 - any failure just means the default font
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dream-exp", default="latent_ppo_pusht_lewm_sparse_dense_correlation")
    parser.add_argument("--seed", type=int, default=1, help="training seed of the run to inspect")
    parser.add_argument("--checkpoint", default=None, help="explicit checkpoint (overrides --seed lookup)")
    parser.add_argument("--variant", default="final", help="checkpoint name inside the run dir")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/rq2/success_gallery")
    parser.add_argument("--episodes", type=int, default=96, help="dream episodes to roll out")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=12345)
    parser.add_argument("--max-tiles", type=int, default=24, help="declared successes on the contact sheet")
    parser.add_argument("--strip-steps", type=int, default=6, help="frames per per-episode filmstrip")
    parser.add_argument("--max-strips", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--verify-in-sim",
        action="store_true",
        help="replay each declared success in the real simulator to measure the hallucination rate",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ dream side
def build_world_and_agent(args):
    """Rebuild the exact dream world the checkpoint was trained in."""
    from src.evaluation.agents import load_ppo_components, resolve_device
    from src.ppo.train_lewm import DreamConfig, LeWMDreamWorld

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        run = find_run_dir(args.dream_exp, args.seed, args.runs_root)
        checkpoint = run.path / f"{args.variant}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    print(f"inspecting {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})
    if "wm_frameskip" not in saved:
        raise SystemExit(
            f"{checkpoint} was not produced by src.ppo.train_lewm; this script "
            "inspects dream-PPO runs only."
        )
    # Derived config fields (batch_size, num_iterations, ...) are init=False.
    init_names = {field.name for field in fields(DreamConfig) if field.init}
    cfg = DreamConfig(**{key: value for key, value in saved.items() if key in init_names})
    cfg.num_envs = args.num_envs

    device = resolve_device(args.device)
    world = LeWMDreamWorld(cfg, device)
    # This script exists to look at the imagined frames, so it always needs them
    # decoded -- including for a de-projector run, whose policy path never
    # renders anything.
    world.capture_frames = True
    # The policy must consume exactly the latents the dream world produces, from
    # the same frozen ViT, as it did during training.
    components = load_ppo_components(checkpoint, device, encoder=world.cls_encoder)
    return cfg, world, components.agent, device, checkpoint


@torch.no_grad()
def collect_dream_episodes(cfg, world, agent, args) -> list[dict]:
    """Deterministic dream rollouts from held-out anchors, recording everything."""
    from src.ppo.env import LatentHistory

    horizon = cfg.dream_eval_steps or cfg.dream_episode_steps
    n = cfg.num_envs
    quota = np.full(n, args.episodes // n, dtype=np.int64)
    quota[: args.episodes % n] += 1

    # Only the episodes that get rendered need their frames kept. A 20-step
    # episode is ~12 MB of float32 RGB, so retaining all of them would cost more
    # than a gigabyte at the default episode count for no benefit.
    keep_frames_for = max(args.max_tiles, args.max_strips)
    episodes: list[dict] = []
    with world.eval_mode(args.rollout_seed, horizon):
        histories = [LatentHistory(cfg.frame_stack, 1) for _ in range(n)]
        live = [None] * n

        def restart(i):
            context = world.reset_env(i)
            histories[i].clear()
            for t in range(context.shape[0]):
                histories[i].append(context[t])
            live[i] = {
                "anchor": dict(world.last_anchor[i]),
                "frames": [],
                "actions": [],
                "success_prob": [],
            }

        for i in range(n):
            restart(i)

        while quota.sum() > 0:
            stacked = torch.stack([history.stacked() for history in histories], dim=0)
            # The mean action: this is the policy that would be deployed.
            action = torch.clamp(agent.actor.bc_policy(stacked), -1.0, 1.0)
            cls, _, terminated, truncated, _ = world.step(action)

            frames = to_uint8(world.last_frames.detach().cpu())
            probs = (
                world.last_success_prob.detach().cpu().numpy()
                if world.last_success_prob is not None
                else np.full(n, np.nan)
            )
            for i in range(n):
                live[i]["frames"].append(frames[i])
                live[i]["actions"].append(action[i].detach().cpu().numpy())
                live[i]["success_prob"].append(float(probs[i]))
                histories[i].append(cls[i])

                if not (terminated[i] or truncated[i]):
                    continue
                if quota[i] > 0:
                    episode = {
                        **live[i],
                        "env": i,
                        "declared_success": bool(terminated[i]),
                        "steps": len(live[i]["frames"]),
                    }
                    rendered = sum(item["declared_success"] for item in episodes)
                    if not episode["declared_success"] or rendered >= keep_frames_for:
                        # Actions and probabilities stay: the simulator check and
                        # the summary statistics need every declared success.
                        episode["frames"] = []
                    episodes.append(episode)
                    quota[i] -= 1
                restart(i)

    declared = sum(episode["declared_success"] for episode in episodes)
    print(
        f"rolled {len(episodes)} dream episodes | {declared} terminated as "
        f"probe-declared successes ({declared / max(len(episodes), 1):.1%})"
    )
    return episodes


# -------------------------------------------------------------- simulator side
def verify_in_simulator(episodes, cfg, args) -> dict:
    """Replay each declared success from its anchor state in the real env.

    The dream episode was seeded from a known dataset frame, so the simulator can
    be put into that exact state and driven with the exact chunks imagination
    produced. Whatever the simulator then reports is the ground truth the probe
    was supposed to predict.
    """
    import h5py

    from src.envs import make_pusht_env

    horizon_env_steps = (cfg.dream_eval_steps or cfg.dream_episode_steps) * cfg.wm_frameskip
    env = make_pusht_env(
        env_id=cfg.env_id,
        max_episode_steps=horizon_env_steps + cfg.wm_frameskip,
        align_sampled_goal_to_fixed_target=True,
        fixed_target_block_success=True,
        block_start_near_goal=False,
    )
    unwrapped = env.unwrapped
    pixel_errors = []
    verified = 0
    checked = 0

    try:
        with h5py.File(repo_path(cfg.dataset_path), "r") as h5:
            for episode in episodes:
                if not episode["declared_success"]:
                    continue
                checked += 1
                env.reset(seed=args.rollout_seed + checked)
                state = np.asarray(episode["anchor"]["state"], dtype=np.float64)
                full_state = np.asarray(unwrapped._get_obs(), dtype=np.float64).copy()
                full_state[: len(state)] = state
                unwrapped._set_state(full_state)

                # Sanity: the simulator frame should match the dataset frame the
                # dream episode started from. A large error means the state
                # mapping is wrong and the verdicts below cannot be trusted.
                rendered = np.asarray(env.render(), dtype=np.float32)
                reference = h5["pixels"][episode["anchor"]["row"]].astype(np.float32)
                if rendered.shape == reference.shape:
                    pixel_errors.append(float(np.abs(rendered - reference).mean()))

                success = False
                terminated = truncated = False
                for chunk in episode["actions"]:
                    for step in range(cfg.wm_frameskip):
                        _, _, terminated, truncated, info = env.step(
                            np.asarray(chunk[step], dtype=np.float32)
                        )
                        success = success or bool(info.get("block_success", terminated))
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        break
                episode["real_success"] = bool(success)
                verified += bool(success)
    finally:
        env.close()

    stats = {
        "declared_successes_checked": checked,
        "confirmed_in_simulator": verified,
        "hallucinated": checked - verified,
        "hallucination_rate": (checked - verified) / checked if checked else float("nan"),
        "mean_anchor_pixel_error": float(np.mean(pixel_errors)) if pixel_errors else None,
    }
    print(
        f"simulator check | {verified}/{checked} declared successes confirmed "
        f"-> hallucination rate {stats['hallucination_rate']:.1%}"
    )
    if stats["mean_anchor_pixel_error"] is not None and stats["mean_anchor_pixel_error"] > 12.0:
        print(
            f"WARNING: mean anchor pixel error {stats['mean_anchor_pixel_error']:.1f}/255 is high; "
            "the simulator may not be starting from the dataset state. Treat the "
            "hallucination rate as unverified."
        )
    return stats


# ------------------------------------------------------------------- rendering
def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``[..., 3, H, W]`` float RGB in [0, 1] -> ``[..., H, W, 3]`` uint8.

    Converted at capture time: uint8 is a quarter of the memory, and the frames
    are only ever written out as images.
    """
    array = frames.detach().cpu().clamp(0, 1).movedim(-3, -1).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def _verdict(episode) -> tuple[str, str]:
    """Status slot plus its label -- a status colour never carries meaning alone."""
    if "real_success" not in episode:
        return "unknown", "probe says SUCCESS (unverified)"
    if episode["real_success"]:
        return "good", "CONFIRMED in simulator"
    return "critical", "HALLUCINATED"


def contact_sheet(episodes, path: Path, args, frameskip: int, verification=None) -> None:
    """Terminal imagined frames of probe-declared successes, in one sheet."""
    declared = [episode for episode in episodes if episode["declared_success"]]
    if not declared:
        print("no probe-declared successes to render")
        return
    declared = declared[: args.max_tiles]

    tile = 180
    columns = min(5, len(declared))
    rows = (len(declared) + columns - 1) // columns
    # Caption holds three short lines rather than one long one: at 180px a tile
    # is narrower than "probe p=1.00 · after 60 imagined env steps".
    pad, header, caption = 12, 62, 56
    title_font, label_font, meta_font = load_font(19), load_font(13), load_font(12)

    width = columns * tile + (columns + 1) * pad
    height = header + rows * (tile + caption + pad) + pad
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), "Imagined states PPO scored as task success", fill=INK_PRIMARY, font=title_font)
    subtitle = "grey T = imagined block · green T = goal"
    if verification:
        subtitle = (
            f"{verification['hallucinated']} of "
            f"{verification['declared_successes_checked']} did not happen when the same "
            f"actions were replayed in the simulator · {subtitle}"
        )
    draw.text((pad, 36), subtitle, fill=INK_SECONDARY, font=meta_font)

    for index, episode in enumerate(declared):
        row, column = divmod(index, columns)
        x = pad + column * (tile + pad)
        y = header + row * (tile + caption + pad)
        image = Image.fromarray(episode["frames"][-1]).resize((tile, tile))
        canvas.paste(image, (x, y))
        status, label = _verdict(episode)
        color = STATUS_COLORS[status]
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=color, width=3)
        draw.text((x, y + tile + 5), label, fill=color, font=label_font)
        draw.text(
            (x, y + tile + 22),
            f"probe p={episode['success_prob'][-1]:.2f}",
            fill=INK_SECONDARY,
            font=meta_font,
        )
        draw.text(
            (x, y + tile + 37),
            f"after {episode['steps'] * frameskip} imagined env steps",
            fill=INK_SECONDARY,
            font=meta_font,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f"wrote {path}")


def filmstrips(episodes, output_dir: Path, args, frameskip: int) -> None:
    """Per-episode imagined trajectory, ending on the declared-success frame."""
    declared = [episode for episode in episodes if episode["declared_success"]][: args.max_strips]
    output_dir.mkdir(parents=True, exist_ok=True)
    title_font, meta_font = load_font(17), load_font(12)
    for index, episode in enumerate(declared):
        frames = episode["frames"]
        picks = np.unique(np.linspace(0, len(frames) - 1, args.strip_steps).astype(int))
        pad, header, caption = 8, 34, 26
        width = len(picks) * TILE + (len(picks) + 1) * pad
        height = header + TILE + caption + 2 * pad
        canvas = Image.new("RGB", (width, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        status, label = _verdict(episode)
        draw.text(
            (pad, 9),
            f"dream episode {index} — {label}",
            fill=STATUS_COLORS[status],
            font=title_font,
        )
        for column, pick in enumerate(picks):
            x = pad + column * (TILE + pad)
            canvas.paste(Image.fromarray(frames[pick]), (x, header + pad))
            draw.text(
                (x, header + pad + TILE + 5),
                f"+{(pick + 1) * frameskip} env steps · probe p={episode['success_prob'][pick]:.2f}",
                fill=INK_SECONDARY,
                font=meta_font,
            )
        canvas.save(output_dir / f"episode_{index:02d}.png")
    if declared:
        print(f"wrote {len(declared)} filmstrips to {output_dir}")


def main() -> None:
    args = parse_args()
    output_root = repo_path(args.output_root) / f"seed{args.seed}_{args.variant}"

    cfg, world, agent, _device, checkpoint = build_world_and_agent(args)
    try:
        episodes = collect_dream_episodes(cfg, world, agent, args)
        verification = verify_in_simulator(episodes, cfg, args) if args.verify_in_sim else None
    finally:
        world.close()

    contact_sheet(
        episodes, output_root / "declared_successes.png", args, cfg.wm_frameskip, verification
    )
    filmstrips(episodes, output_root / "filmstrips", args, cfg.wm_frameskip)

    declared = [episode for episode in episodes if episode["declared_success"]]
    summary = {
        "checkpoint": str(checkpoint),
        "episodes": len(episodes),
        "declared_successes": len(declared),
        "declared_success_rate": len(declared) / max(len(episodes), 1),
        "mean_declared_probability": (
            float(np.mean([episode["success_prob"][-1] for episode in declared]))
            if declared
            else None
        ),
        "mean_steps_to_declared_success": (
            float(np.mean([episode["steps"] for episode in declared])) if declared else None
        ),
        "horizon_predictor_steps": cfg.dream_eval_steps or cfg.dream_episode_steps,
        "wm_frameskip": cfg.wm_frameskip,
        "rollout_seed": args.rollout_seed,
        "verification": verification,
        # Per-episode records, so the hallucination rate can be read against the
        # step the probe fired at. A pooled rate cannot say whether shortening
        # dream_episode_steps would remove the false successes or leave them.
        "records": [
            {
                "steps_to_declared_success": int(episode["steps"]),
                "declared_probability": float(episode["success_prob"][-1]),
                "real_success": episode.get("real_success"),
            }
            for episode in declared
        ],
    }
    path = output_root / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
