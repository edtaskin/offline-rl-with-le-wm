"""Compatibility entrypoint for PPO evaluation.

New code should use ``python -m src.evaluation.evaluate_pusht --agent-type ppo``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.envs import PUSHT_FIXED_TARGET_POSE
from src.evaluation.agents import (
    load_ppo_components,
    make_ppo_evaluation_agent,
)
from src.evaluation.pusht import PushTEvalConfig, run_evaluation


def load_latent_agent(checkpoint, device):
    components = load_ppo_components(checkpoint, device)
    return components.agent, components.encoder, components.config, components.contract


def evaluate(
    checkpoint,
    episodes=20,
    deterministic=True,
    record_video=False,
    video_dir=None,
    video_resolution=512,
    replan_interval=0,
    temporal_ensemble=False,
    temporal_ensemble_decay=0.01,
    fixed_target_pose=None,
    fixed_target_block_success=None,
    seed=42,
    device="auto",
    max_episode_steps=300,
    block_start_radius=None,
):
    if temporal_ensemble and not deterministic:
        raise ValueError("temporal ensembling requires deterministic chunk predictions")
    components = load_ppo_components(checkpoint, device)
    chunk_size = int(components.contract["action_chunk_size"])
    if temporal_ensemble:
        execution_mode = "temporal-ensemble"
    elif replan_interval and replan_interval < chunk_size:
        execution_mode = "receding-horizon"
    else:
        execution_mode = "open-loop"
    adapter = make_ppo_evaluation_agent(
        checkpoint=checkpoint,
        components=components,
        deterministic=deterministic,
        execution_mode=execution_mode,
        replan_interval=max(1, replan_interval),
        temporal_ensemble_decay=temporal_ensemble_decay,
    )
    result = run_evaluation(
        adapter,
        PushTEvalConfig(
            episodes=episodes,
            seed=seed,
            max_episode_steps=max_episode_steps,
            fixed_target_pose=tuple(fixed_target_pose or PUSHT_FIXED_TARGET_POSE.tolist()),
            fixed_target_block_success=(
                True if fixed_target_block_success is None else fixed_target_block_success
            ),
            block_start_radius=block_start_radius,
            record_video=record_video,
            video_dir=video_dir or str(Path(checkpoint).with_suffix("")),
            video_resolution=video_resolution,
        ),
    )
    return result.summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent PPO on canonical PushT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--fixed-target-pose", type=float, nargs=3, default=None)
    parser.add_argument(
        "--fixed-target-block-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--block-start-radius",
        type=float,
        default=None,
        help="sample block starts within this goal radius; omit for unrestricted starts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
        fixed_target_pose=args.fixed_target_pose,
        fixed_target_block_success=args.fixed_target_block_success,
        seed=args.seed,
        device=args.device,
        max_episode_steps=args.max_episode_steps,
        block_start_radius=args.block_start_radius,
    )


if __name__ == "__main__":
    main()
