"""Check that the offline rollout helper reproduces the live dream world exactly.

``scripts/deprojector/common.imagine`` re-implements the predictor loop so the
de-projector can be trained and scored offline. If it drifted from
``LeWMDreamWorld.step`` -- context construction, action alignment, history
truncation -- every number the other scripts produce would describe a world the
trainer never rolls in. This drives the real dream world from a known anchor
with ground-truth actions and compares its imagined latents against the offline
path step by step.

Usage::

    python -m scripts.deprojector.verify_parity --steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.deprojector.common import (  # noqa: E402
    encode_cls,
    imagine,
    normalize_blocks,
    open_expert_h5,
    read_anchor_window,
)
from src.ppo.train_lewm import DreamConfig, LeWMDreamWorld  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="predictor steps to compare")
    parser.add_argument("--anchors", type=int, default=4, help="dream resets to check")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max relative deviation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    cfg = DreamConfig(
        exp_name="deprojector_parity",
        num_envs=1,
        frame_stack=3,
        frame_stride=5,
        action_chunk_size=5,
        reward_mode="sparse",
        fixed_target=True,
        block_start_near_goal=True,
        block_start_radius=200.0,
        dream_episode_steps=max(args.steps + 1, 2),
        dream_eval_interval=1,
        eval_interval=0,
        selection="dream",
        seed=args.seed,
    )
    world = LeWMDreamWorld(cfg, device)
    history_size = world.history_size
    context_steps = world.context_steps
    fs = cfg.wm_frameskip

    h5 = open_expert_h5(cfg.dataset_path)
    worst = 0.0
    try:
        for trial in range(args.anchors):
            world.reset_env(0)
            anchor = world.last_anchor[0]["row"]

            # Drive the live world with the dataset's own actions from this anchor.
            live = []
            for step in range(args.steps):
                rows = anchor + step * fs
                block = torch.as_tensor(
                    h5["action"][rows : rows + fs], dtype=torch.float32, device=device
                ).unsqueeze(0)
                world.step(block)
                live.append(world.last_latent[0].clone())
            live = torch.stack(live)

            # Same anchor, offline path.
            frames, blocks = read_anchor_window(
                h5, anchor, context_steps=context_steps, frameskip=fs, horizon=args.steps
            )
            cls = encode_cls(world.cls_encoder, frames, 64, device).to(device)
            with torch.no_grad():
                emb = world.wm.projector(cls[:context_steps]).unsqueeze(0)
                offline = imagine(
                    world.wm,
                    emb,
                    normalize_blocks(
                        torch.as_tensor(blocks, dtype=torch.float32, device=device).unsqueeze(0),
                        world.action_mean,
                        world.action_std,
                    ),
                    horizon=args.steps,
                    history_size=history_size,
                )[0]

            deviation = float(
                (live - offline).norm(dim=-1).max() / offline.norm(dim=-1).max().clamp_min(1e-12)
            )
            worst = max(worst, deviation)
            print(f"anchor {trial + 1}/{args.anchors} (row {anchor}): max relative deviation {deviation:.2e}")
    finally:
        h5.close()
        world.close()

    print(f"\nworst relative deviation over {args.anchors} anchors x {args.steps} steps: {worst:.2e}")
    if worst > args.tolerance:
        print(f"FAIL: offline rollout diverges from LeWMDreamWorld (tolerance {args.tolerance:.0e})")
        return 1
    print(f"OK: offline rollout matches LeWMDreamWorld within {args.tolerance:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
