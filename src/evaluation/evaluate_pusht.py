"""Canonical PushT evaluator for BC and latent PPO checkpoints.

``canonical_v2`` balances in-support initial block poses across
translation/rotation strata. ``canonical_ood`` applies the same evaluation
logic to a feasible 200--260 px annulus outside the training start radius.
Their exact boundaries, formulas, and intended interpretation are documented
in ``src/evaluation/README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.envs import PUSHT_FIXED_TARGET_POSE, PUSHT_RENDER_SHAPE
from src.evaluation.agents import make_bc_evaluation_agent, make_ppo_evaluation_agent
from src.evaluation.baseline_agents import (
    make_cnn_bc_evaluation_agent,
    make_state_bc_evaluation_agent,
)
from src.evaluation.pusht import (
    CANONICAL_OOD,
    CANONICAL_OOD_ANGLE_THRESHOLDS,
    CANONICAL_OOD_COMPLETION_BUDGETS,
    CANONICAL_OOD_DISTANCE_THRESHOLDS,
    CANONICAL_OOD_MAX_RADIUS,
    CANONICAL_OOD_MIN_RADIUS,
    CANONICAL_V1,
    CANONICAL_V2,
    CANONICAL_V2_ANGLE_THRESHOLDS,
    CANONICAL_V2_COMPLETION_BUDGETS,
    CANONICAL_V2_DISTANCE_THRESHOLDS,
    PushTEvalConfig,
    run_evaluation,
    select_stratified_episode_seeds,
)
from src.evaluation.start_visualization import write_start_location_visualization


MIN_EPISODE_SEED_GAP = 7
MAX_EPISODE_SEED = np.iinfo(np.int32).max


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate an agent on the canonical PushT env")
    parser.add_argument(
        "--protocol",
        choices=[CANONICAL_V1, CANONICAL_V2, CANONICAL_OOD],
        default=CANONICAL_V1,
        help=(
            "canonical_v1 preserves the distribution-matched seed suite; "
            "canonical_v2 balances six in-support start strata; canonical_ood "
            "balances six feasible strata in the 200-260px centroid annulus"
        ),
    )
    parser.add_argument(
        "--agent-type",
        choices=["bc", "ppo", "bc-state", "bc-cnn"],
        required=True,
        help="bc/ppo read frozen LeWM latents; bc-state and bc-cnn are the encoder baselines",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="hf:// artifact reference or experimental local path outside checkpoints/",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "explicit local LeWM object checkpoint for latent BC/PPO; "
            "omitting it uses $STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt"
        ),
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
    parser.add_argument(
        "--block-start-min-radius",
        type=float,
        default=None,
        help=(
            "optional inner centroid radius; canonical_ood fixes this at 200 "
            "and samples an annulus rather than a disk"
        ),
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--output-root", default="runs/evaluations")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--capture-traces", action="store_true")
    parser.add_argument(
        "--visualize-starts",
        action="store_true",
        help=(
            "save start_locations.png with every deterministic block start, "
            "orientation, difficulty stratum, and distance threshold"
        ),
    )
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
        "start_locations_path": str(Path(run_dir) / "start_locations.png")
        if result.config.visualize_starts
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
    if args.encoder_checkpoint is not None and args.agent_type not in {"bc", "ppo"}:
        raise ValueError(
            "--encoder-checkpoint is only valid for latent BC/PPO agents"
        )
    agent_kwargs = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "execution_mode": args.execution_mode,
        "replan_interval": args.replan_interval,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
    }
    if args.agent_type == "bc":
        agent = make_bc_evaluation_agent(
            stats_path=args.stats,
            encoder_checkpoint=args.encoder_checkpoint,
            **agent_kwargs,
        )
    elif args.agent_type == "bc-state":
        agent = make_state_bc_evaluation_agent(stats_path=args.stats, **agent_kwargs)
    elif args.agent_type == "bc-cnn":
        agent = make_cnn_bc_evaluation_agent(stats_path=args.stats, **agent_kwargs)
    else:
        agent = make_ppo_evaluation_agent(
            encoder_checkpoint=args.encoder_checkpoint,
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
    block_start_radius = args.block_start_radius
    block_start_min_radius = args.block_start_min_radius
    block_start_clip_out_of_bounds = True
    if args.protocol == CANONICAL_V2:
        if block_start_radius is None:
            block_start_radius = 200.0
        if block_start_min_radius is None:
            block_start_min_radius = 0.0
        if not np.isclose(block_start_radius, 200.0):
            raise ValueError(
                "canonical_v2 is defined for --block-start-radius 200"
            )
    elif args.protocol == CANONICAL_OOD:
        if block_start_radius is None:
            block_start_radius = CANONICAL_OOD_MAX_RADIUS
        if block_start_min_radius is None:
            block_start_min_radius = CANONICAL_OOD_MIN_RADIUS
        block_start_clip_out_of_bounds = False
    elif block_start_min_radius is None:
        block_start_min_radius = 0.0

    if args.protocol == CANONICAL_V2:
        distance_thresholds = CANONICAL_V2_DISTANCE_THRESHOLDS
        angle_thresholds = CANONICAL_V2_ANGLE_THRESHOLDS
        completion_budgets = CANONICAL_V2_COMPLETION_BUDGETS
    elif args.protocol == CANONICAL_OOD:
        distance_thresholds = CANONICAL_OOD_DISTANCE_THRESHOLDS
        angle_thresholds = CANONICAL_OOD_ANGLE_THRESHOLDS
        completion_budgets = CANONICAL_OOD_COMPLETION_BUDGETS
    else:
        distance_thresholds = ()
        angle_thresholds = ()
        completion_budgets = ()

    config = PushTEvalConfig(
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        protocol=args.protocol,
        max_episode_steps=args.max_episode_steps,
        observation_resolution=args.observation_resolution,
        fixed_target_pose=tuple(args.fixed_target_pose),
        fixed_target_block_success=args.fixed_target_block_success,
        fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
        agent_block_coef=args.agent_block_coef,
        block_start_radius=block_start_radius,
        block_start_min_radius=block_start_min_radius,
        block_start_clip_out_of_bounds=block_start_clip_out_of_bounds,
        record_video=args.video,
        video_fps=args.video_fps,
        video_resolution=args.video_resolution,
        capture_traces=args.capture_traces,
        allow_resolution_mismatch=args.allow_resolution_mismatch,
        visualize_starts=args.visualize_starts,
        distance_thresholds=distance_thresholds,
        angle_thresholds=angle_thresholds,
        completion_budgets=completion_budgets,
    )
    if args.protocol in {CANONICAL_V2, CANONICAL_OOD}:
        candidate_count = (
            max(30_000, args.episodes * 400)
            if args.protocol == CANONICAL_OOD
            else max(10_000, args.episodes * 200)
        )
        candidates = sample_episode_seeds(args.seed, candidate_count)
        suite = select_stratified_episode_seeds(config, candidates)
        config = replace(
            config,
            episode_seeds=suite.seeds,
            episode_strata=suite.strata,
        )
        print(
            f"{args.protocol} suite | "
            f"examined={suite.candidates_examined} | counts={suite.counts}"
        )
    else:
        episode_seeds = sample_episode_seeds(args.seed, args.episodes)
        config = replace(config, episode_seeds=tuple(episode_seeds))

    run_dir = create_run_directory(
        args.output_root,
        args.agent_type,
        args.checkpoint,
        run_name=args.run_name,
    )
    config = replace(config, video_dir=str(run_dir / "videos"))
    print(f"Evaluation run directory: {run_dir}")
    if config.visualize_starts:
        write_start_location_visualization(
            config,
            run_dir / "start_locations.png",
        )
    episode_seeds = config.episode_seeds or ()
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
