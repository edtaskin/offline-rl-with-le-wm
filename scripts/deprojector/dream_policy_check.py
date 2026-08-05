"""Does the dream agree with reality about a policy we already know the answer for?

Runs the published raw-CLS BC policy inside :class:`LeWMDreamWorld` -- the same
world dream PPO would train in -- and reports imagined success. Because that
policy's real success rate is already measured, the gap is a direct read on
whether the dream is faithful enough to optimize in, *before* spending a 1M-step
training run. A large optimism gap here means PPO would be improving a number
that does not transfer, whichever bridge is used.

Scores either bridge, so the de-projector and the decoder are compared on the
one quantity that matters rather than on reconstruction error.

Usage::

    python -m scripts.deprojector.dream_policy_check --bridge both --episodes 96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.deprojector.bridge_horizon import load_raw_cls_bc_policy  # noqa: E402
from scripts.deprojector.common import repo_path  # noqa: E402
from src.ppo.env import LatentHistory  # noqa: E402
from src.ppo.train_lewm import DreamConfig, LeWMDreamWorld  # noqa: E402
from src.representations.deprojector import load_deprojector  # noqa: E402

DEFAULT_BC = "hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth"
DEFAULT_BC_STATS = "hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best_stats.pth"


class DeprojectorDreamWorld(LeWMDreamWorld):
    """Dream world whose bridge is a learned latent map instead of the decoder."""

    def __init__(self, cfg, device, deprojector_path):
        super().__init__(cfg, device)
        self.deprojector = load_deprojector(repo_path(deprojector_path), device)

    def _observe(self, pred: torch.Tensor) -> torch.Tensor:
        return self.deprojector(pred)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", choices=("deprojector", "decoder", "both"), default="both")
    parser.add_argument("--deprojector", default="models/deprojector/pusht_lewm/deprojector.pt")
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC)
    parser.add_argument("--bc-stats", default=DEFAULT_BC_STATS)
    parser.add_argument("--episodes", type=int, default=96, help="imagined episodes per bridge")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--dream-episode-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--block-start-radius", type=float, default=200.0)
    parser.add_argument(
        "--real-success",
        type=float,
        default=None,
        help="the same policy's measured real success rate, for the optimism gap",
    )
    parser.add_argument("--output-dir", default="runs/deprojector")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def build_config(args):
    """Dream config matching the RQ1 runs, minus anything the world does not need."""
    return DreamConfig(
        exp_name="deprojector_dream_policy_check",
        num_envs=args.num_envs,
        frame_stack=3,
        frame_stride=5,
        action_chunk_size=5,
        reward_mode="sparse",
        fixed_target=True,
        block_start_near_goal=True,
        block_start_radius=args.block_start_radius,
        dream_episode_steps=args.dream_episode_steps,
        dream_eval_interval=1,
        eval_interval=0,
        selection="dream",
        seed=args.seed,
    )


@torch.no_grad()
def imagined_success(world, policy, cfg, *, episodes, seed, horizon, device):
    """Fixed per-env episode quota, mirroring ``LeWMDreamPPOTrainer._evaluate_dream``.

    Sampling until N episodes merely *finish* would over-count short ones, and
    imagined episodes end early exactly when the probe declares success.
    """
    n = cfg.num_envs
    quota = np.full(n, episodes // n, dtype=np.int64)
    quota[: episodes % n] += 1

    successes, lengths = [], []
    with world.eval_mode(seed, horizon):
        histories = [LatentHistory(cfg.frame_stack, 1) for _ in range(n)]
        for i in range(n):
            ctx = world.reset_env(i)
            for t in range(ctx.shape[0]):
                histories[i].append(ctx[t])
        steps = np.zeros(n, dtype=np.int64)

        while quota.sum() > 0:
            stacked = torch.stack([h.stacked() for h in histories], dim=0).to(device)
            action = torch.clamp(policy(stacked), -1.0, 1.0)
            cls, _, terminated, truncated, _ = world.step(action)
            steps += 1
            for i in range(n):
                histories[i].append(cls[i])
                if not (terminated[i] or truncated[i]):
                    continue
                if quota[i] > 0:
                    successes.append(float(terminated[i]))
                    lengths.append(float(steps[i]))
                    quota[i] -= 1
                steps[i] = 0
                ctx = world.reset_env(i)
                histories[i].clear()
                for t in range(ctx.shape[0]):
                    histories[i].append(ctx[t])

    return {
        "imagined_success_rate": float(np.mean(successes)),
        "mean_length_steps": float(np.mean(lengths)),
        "mean_length_env_steps": float(np.mean(lengths)) * cfg.wm_frameskip,
        "episodes": len(successes),
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    cfg = build_config(args)
    policy, _ = load_raw_cls_bc_policy(args.bc_checkpoint, args.bc_stats, device)

    bridges = ["deprojector", "decoder"] if args.bridge == "both" else [args.bridge]
    results = {}
    for bridge in bridges:
        print(f"\n=== bridge: {bridge} ===", flush=True)
        if bridge == "deprojector":
            world = DeprojectorDreamWorld(cfg, device, args.deprojector)
        else:
            world = LeWMDreamWorld(cfg, device)
        try:
            stats = imagined_success(
                world,
                policy,
                cfg,
                episodes=args.episodes,
                seed=args.seed,
                horizon=args.dream_episode_steps,
                device=device,
            )
        finally:
            world.close()
        results[bridge] = stats
        line = (
            f"{bridge:12s} imagined success {stats['imagined_success_rate']:.3f} "
            f"over {stats['episodes']} episodes | mean length {stats['mean_length_steps']:.1f} "
            f"predictor steps ({stats['mean_length_env_steps']:.0f} env steps)"
        )
        if args.real_success is not None:
            gap = stats["imagined_success_rate"] - args.real_success
            stats["real_success_rate"] = args.real_success
            stats["optimism_gap"] = gap
            line += f" | real {args.real_success:.3f} | optimism gap {gap:+.3f}"
        print(line, flush=True)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dream_policy_check.json"
    path.write_text(
        json.dumps({"args": vars(args), "results": results}, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
