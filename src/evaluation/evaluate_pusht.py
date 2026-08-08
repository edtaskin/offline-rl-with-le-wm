"""Canonical PushT evaluator for BC and latent PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.envs import PUSHT_FIXED_TARGET_POSE, PUSHT_RENDER_SHAPE
from src.evaluation.agents import make_bc_evaluation_agent, make_ppo_evaluation_agent
from src.evaluation.baseline_agents import make_state_bc_evaluation_agent
from src.evaluation.pusht import (
    PushTEvalConfig,
    run_evaluation,
)


MIN_EPISODE_SEED_GAP = 7
MAX_EPISODE_SEED = np.iinfo(np.int32).max


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate an agent on the canonical PushT env")
    parser.add_argument(
        "--agent-type",
        choices=["bc", "ppo", "bc-state"],
        required=True,
        help="bc/ppo read frozen LeWM latents; bc-state is the encoder-baseline oracle",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="hf:// artifact reference or experimental local path outside checkpoints/",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="BC stats artifact; inferred from the BC checkpoint when omitted",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample PPO actions instead of using the mean")
    parser.add_argument(
        "--execution-mode",
        choices=["open-loop", "receding-horizon", "temporal-ensemble"],
        default="open-loop",
    )
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)

    parser.add_argument("--env-id", default="swm/PushT-v1")
    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="total number of evaluation episodes (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="master seed used to sample reproducible, well-separated episode seeds",
    )
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--observation-resolution",
        type=int,
        default=PUSHT_RENDER_SHAPE[0],
        help="square policy-observation resolution (default: 224)",
    )
    parser.add_argument(
        "--training-observation-resolution",
        type=int,
        default=None,
        help="native training resolution for a legacy checkpoint that does not record it",
    )
    parser.add_argument(
        "--allow-resolution-mismatch",
        action="store_true",
        help="intentionally evaluate at a resolution different from training",
    )
    parser.add_argument(
        "--fixed-target-pose",
        type=float,
        nargs=3,
        default=PUSHT_FIXED_TARGET_POSE.tolist(),
        metavar=("X", "Y", "ANGLE"),
    )
    parser.add_argument(
        "--fixed-target-block-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-target-max-reset-attempts", type=int, default=100)
    parser.add_argument("--agent-block-coef", type=float, default=0.0)
    parser.add_argument(
        "--block-start-radius",
        type=float,
        default=None,
        help="sample block starts within this goal radius; omit for unrestricted starts",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--output-root", default="runs/evaluations")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--capture-traces", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="offline-rl-lewm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser


def _slug(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return slug or "eval"


def sample_episode_seeds(
    seed,
    episodes,
    min_gap=MIN_EPISODE_SEED_GAP,
    max_seed=MAX_EPISODE_SEED,
):
    """Sample reproducible, unique episode seeds from one master seed.

    Seeds are sampled in a large integer space rather than taken consecutively.
    Sampling unique slots on a ``min_gap`` grid guarantees that every pair of
    returned seed values differs by at least ``min_gap``. This avoids the known
    adjacent-seed collisions in the underlying PushT reset while preserving an
    exactly reproducible evaluation suite.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if min_gap < 1:
        raise ValueError("min_gap must be at least 1")
    if max_seed < 0:
        raise ValueError("max_seed must be non-negative")

    slot_count = max_seed // min_gap + 1
    if episodes > slot_count:
        raise ValueError("episodes exceed the available well-separated seed space")

    rng = np.random.default_rng(seed)
    slots = set()
    episode_seeds = []
    while len(episode_seeds) < episodes:
        slot = int(rng.integers(0, slot_count))
        if slot in slots:
            continue
        slots.add(slot)
        episode_seeds.append(slot * min_gap)
    return episode_seeds


def create_run_directory(
    output_root,
    agent_type,
    checkpoint,
    run_name=None,
    timestamp=None,
):
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, _slug(agent_type), _slug(Path(checkpoint).stem)]
    if run_name:
        parts.append(_slug(run_name))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = "_".join(parts)
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        run_dir = output_root / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"could not allocate an evaluation run directory under {output_root}")


def _write_metrics(result, run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
    payload = result.to_dict()
    payload["artifacts"] = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "video_dir": str(Path(run_dir) / "videos")
        if result.config.record_video
        else None,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Saved evaluation metrics to: {metrics_path}")
    return metrics_path


def evaluate_from_args(args):
    if args.agent_type != "ppo" and args.stochastic:
        raise ValueError("BC evaluation is deterministic; --stochastic is only valid for PPO")
    if args.stochastic and args.execution_mode == "temporal-ensemble":
        raise ValueError("temporal ensembling requires deterministic chunk predictions")
    episode_seeds = sample_episode_seeds(args.seed, args.episodes)
    agent_kwargs = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "execution_mode": args.execution_mode,
        "replan_interval": args.replan_interval,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
    }
    if args.agent_type == "bc":
        agent = make_bc_evaluation_agent(stats_path=args.stats, **agent_kwargs)
    elif args.agent_type == "bc-state":
        agent = make_state_bc_evaluation_agent(stats_path=args.stats, **agent_kwargs)
    else:
        agent = make_ppo_evaluation_agent(
            deterministic=not args.stochastic,
            **agent_kwargs,
        )
    recorded_resolution = agent.metadata.get("training_observation_resolution")
    supplied_resolution = args.training_observation_resolution
    if supplied_resolution is not None and supplied_resolution < 1:
        raise ValueError("training_observation_resolution must be positive")
    if (
        recorded_resolution is not None
        and supplied_resolution is not None
        and int(recorded_resolution) != int(supplied_resolution)
    ):
        raise ValueError(
            "--training-observation-resolution conflicts with checkpoint metadata: "
            f"argument={supplied_resolution}, checkpoint={recorded_resolution}"
        )
    training_resolution = recorded_resolution or supplied_resolution
    if training_resolution is None:
        raise ValueError(
            "checkpoint does not record its training observation resolution; pass "
            "--training-observation-resolution for this legacy checkpoint"
        )
    agent.metadata["training_observation_resolution"] = int(training_resolution)
    if (
        int(training_resolution) != int(args.observation_resolution)
        and not args.allow_resolution_mismatch
    ):
        raise ValueError(
            "evaluation observation resolution does not match model training: "
            f"model={int(training_resolution)}, evaluation={args.observation_resolution}. "
            "Pass the model's training resolution, or use "
            "--allow-resolution-mismatch for an intentional transfer experiment."
        )
    run_dir = create_run_directory(
        args.output_root,
        args.agent_type,
        args.checkpoint,
        run_name=args.run_name,
    )
    print(f"Evaluation run directory: {run_dir}")
    config = PushTEvalConfig(
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        episode_seeds=tuple(episode_seeds),
        max_episode_steps=args.max_episode_steps,
        observation_resolution=args.observation_resolution,
        fixed_target_pose=tuple(args.fixed_target_pose),
        fixed_target_block_success=args.fixed_target_block_success,
        fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
        agent_block_coef=args.agent_block_coef,
        block_start_radius=args.block_start_radius,
        record_video=args.video,
        video_dir=str(run_dir / "videos"),
        video_fps=args.video_fps,
        video_resolution=args.video_resolution,
        capture_traces=args.capture_traces,
        allow_resolution_mismatch=args.allow_resolution_mismatch,
    )
    preview = ", ".join(str(seed) for seed in episode_seeds[:5])
    if len(episode_seeds) > 5:
        preview += ", ..."
    print(
        f"Single evaluation | agent={args.agent_type} | episodes={config.episodes} | "
        f"master_seed={config.seed} | episode_seeds=[{preview}]"
    )
    result = run_evaluation(agent, config)
    return _save_and_track_result(args, result, run_dir)


def _save_and_track_result(args, result, run_dir):
    metrics_path = _write_metrics(result, run_dir)
    # Expose where the metrics landed so programmatic callers (the RQ campaign
    # scripts) can reference the artifact they just produced.
    result.run_dir = str(run_dir)
    result.metrics_path = str(metrics_path)
    if args.wandb:
        import wandb

        init_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "config": result.to_dict()["config"] | {"agent": result.agent_metadata},
        }
        if args.wandb_mode is not None:
            init_kwargs["mode"] = args.wandb_mode
        run = wandb.init(**init_kwargs)
        for step, episode in enumerate(result.episodes, start=1):
            run.log(
                {
                    "eval/episode_return": episode.episode_return,
                    "eval/episode_length": episode.length,
                    "eval/episode_success": episode.success,
                },
                step=step,
            )
        for key, value in result.summary.items():
            run.summary[f"eval/{key}"] = value
        run.finish()
    return result


def main():
    evaluate_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
