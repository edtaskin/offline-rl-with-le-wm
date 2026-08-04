"""Canonical PushT evaluator for BC and latent PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.envs import PUSHT_FIXED_TARGET_POSE, PUSHT_RENDER_SHAPE
from src.evaluation.agents import make_bc_evaluation_agent, make_ppo_evaluation_agent
from src.evaluation.pusht import (
    PushTEvalConfig,
    make_repeat_seeds,
    run_repeated_evaluation,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate an agent on the canonical PushT env")
    parser.add_argument("--agent-type", choices=["bc", "ppo"], required=True)
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
        default=50,
        help="number of episodes per evaluation repeat (default: 50)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of evaluation repeats with non-overlapping seed ranges (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="gap between consecutive episode seeds; use >=7 with unrestricted "
        "block starts, where consecutive seeds collide (see PushTEvalConfig)",
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


def make_repeat_seeds(seed, repeats, episodes, stride=1):
    """Derive deterministic, non-overlapping episode-seed ranges."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return [seed + repeat * episodes * stride for repeat in range(repeats)]


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
    if args.agent_type == "bc" and args.stochastic:
        raise ValueError("BC evaluation is deterministic; --stochastic is only valid for PPO")
    if args.stochastic and args.execution_mode == "temporal-ensemble":
        raise ValueError("temporal ensembling requires deterministic chunk predictions")
    repeat_seeds = make_repeat_seeds(args.seed, args.repeats, args.episodes, args.seed_stride)
    agent_kwargs = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "execution_mode": args.execution_mode,
        "replan_interval": args.replan_interval,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
    }
    if args.agent_type == "bc":
        agent = make_bc_evaluation_agent(stats_path=args.stats, **agent_kwargs)
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
    results = []
    for repeat, repeat_seed in enumerate(repeat_seeds):
        video_dir = run_dir / "videos" / f"repeat_{repeat:02d}_seed_{repeat_seed}"
        config = PushTEvalConfig(
            env_id=args.env_id,
            episodes=args.episodes,
            seed=repeat_seed,
            seed_stride=args.seed_stride,
            max_episode_steps=args.max_episode_steps,
            fixed_target_pose=tuple(args.fixed_target_pose),
            fixed_target_block_success=args.fixed_target_block_success,
            fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
            agent_block_coef=args.agent_block_coef,
            block_start_radius=args.block_start_radius,
            record_video=args.video,
            video_dir=str(video_dir),
            video_fps=args.video_fps,
            video_resolution=args.video_resolution,
            capture_traces=args.capture_traces,
        )
        print(
            f"Repeat {repeat + 1}/{args.repeats} | agent={args.agent_type} | "
            f"fixed-target episodes={config.episodes} | "
            f"seeds={config.seed}..{config.seed + (config.episodes - 1) * config.seed_stride}"
        )
        results.append(run_evaluation(agent, config))
    result = aggregate_evaluation_results(results)
    print("Aggregate evaluation summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value}")
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
