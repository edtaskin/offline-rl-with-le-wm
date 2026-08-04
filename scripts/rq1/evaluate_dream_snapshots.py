"""Evaluate every saved snapshot from one Dream-PPO training run.

The input is the timestamped run directory produced by
``python -m src.ppo.train_lewm``. Every
``snapshot_step<env_steps>_it<iteration>.pt`` is evaluated in the real PushT
environment with the canonical deterministic protocol. Environment identity,
episode horizon, fixed-target success semantics, and the near-goal block-start
radius are inferred from the checkpoint unless explicitly overridden.

Results are written after every snapshot, so an interrupted invocation can be
resumed by running the same command again. A frozen LeWM encoder is shared by
all snapshots from the run; only the small PPO heads are reloaded.

Example::

    python -m scripts.rq1.evaluate_dream_snapshots \
      runs/latent_ppo_pusht_lewm_sparse_dense_correlation__seed1/03082026-200038 \
      --episodes 50 --repeats 3 --eval-seed 42 --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rq_common import (
    CANONICAL_EVAL,
    checkpoint_budget,
    read_jsonl,
    snapshot_checkpoints,
    upsert_jsonl,
)
from src.envs import PUSHT_FIXED_TARGET_POSE


RESULT_KEY = ("checkpoint", "evaluation_id")


@dataclass(frozen=True)
class EvaluationSettings:
    """Checkpoint-independent settings that identify one evaluation protocol."""

    episodes: int
    repeats: int
    seed: int
    seed_stride: int
    env_id: str
    max_episode_steps: int
    fixed_target_pose: tuple[float, float, float]
    fixed_target_block_success: bool
    fixed_target_max_reset_attempts: int
    agent_block_coef: float
    block_start_radius: float | None
    deterministic: bool
    execution_mode: str
    replan_interval: int
    temporal_ensemble_decay: float
    record_video: bool
    video_fps: int
    video_resolution: int
    capture_traces: bool

    @property
    def evaluation_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="timestamped Dream-PPO run directory containing snapshot_*.pt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <run-dir>/snapshot_evaluations)",
    )
    parser.add_argument("--episodes", type=int, default=CANONICAL_EVAL["episodes"])
    parser.add_argument("--repeats", type=int, default=CANONICAL_EVAL["repeats"])
    parser.add_argument("--eval-seed", type=int, default=CANONICAL_EVAL["seed"])
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="gap between episode seeds; use >=7 for unrestricted block starts",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="override the environment recorded in the checkpoint",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="override the episode horizon recorded in the checkpoint",
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
        default=None,
        help="override the checkpoint's fixed-target success semantics",
    )
    parser.add_argument("--fixed-target-max-reset-attempts", type=int, default=100)
    parser.add_argument("--agent-block-coef", type=float, default=0.0)
    block_start = parser.add_mutually_exclusive_group()
    block_start.add_argument(
        "--block-start-radius",
        type=float,
        default=None,
        help="override the checkpoint's near-goal block-start radius",
    )
    block_start.add_argument(
        "--unrestricted-block-start",
        action="store_true",
        help="disable the checkpoint's near-goal block-start distribution",
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=["open-loop", "receding-horizon", "temporal-ensemble"],
        default="open-loop",
    )
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--capture-traces", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="repeat snapshots already scored with exactly this protocol",
    )
    return parser


def resolve_settings(args: argparse.Namespace, checkpoint_config: dict) -> EvaluationSettings:
    """Combine explicit evaluator flags with task settings saved by training."""

    if args.unrestricted_block_start:
        block_start_radius = None
    elif args.block_start_radius is not None:
        block_start_radius = args.block_start_radius
    elif checkpoint_config.get("block_start_near_goal", False):
        block_start_radius = float(checkpoint_config.get("block_start_radius", 200.0))
    else:
        block_start_radius = None

    fixed_target_block_success = args.fixed_target_block_success
    if fixed_target_block_success is None:
        fixed_target_block_success = bool(
            checkpoint_config.get("fixed_target_block_success", True)
        )

    settings = EvaluationSettings(
        episodes=args.episodes,
        repeats=args.repeats,
        seed=args.eval_seed,
        seed_stride=args.seed_stride,
        env_id=args.env_id or checkpoint_config.get("env_id", "swm/PushT-v1"),
        max_episode_steps=(
            args.max_episode_steps
            if args.max_episode_steps is not None
            else int(
                checkpoint_config.get(
                    "max_episode_steps", CANONICAL_EVAL["max_episode_steps"]
                )
            )
        ),
        fixed_target_pose=tuple(float(value) for value in args.fixed_target_pose),
        fixed_target_block_success=fixed_target_block_success,
        fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
        agent_block_coef=args.agent_block_coef,
        block_start_radius=block_start_radius,
        deterministic=not args.stochastic,
        execution_mode=args.execution_mode,
        replan_interval=args.replan_interval,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        record_video=args.video,
        video_fps=args.video_fps,
        video_resolution=args.video_resolution,
        capture_traces=args.capture_traces,
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: EvaluationSettings) -> None:
    if settings.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if settings.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if settings.seed_stride < 1:
        raise ValueError("--seed-stride must be at least 1")
    if settings.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be at least 1")
    if settings.replan_interval < 1:
        raise ValueError("--replan-interval must be at least 1")
    if settings.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay must be non-negative")
    if settings.block_start_radius is not None and settings.block_start_radius < 0:
        raise ValueError("--block-start-radius must be non-negative")
    if settings.record_video and settings.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if not settings.deterministic and settings.execution_mode == "temporal-ensemble":
        raise ValueError("temporal ensembling requires deterministic chunk predictions")


def _is_dream_config(config: dict) -> bool:
    """DreamConfig fields distinguish train_lewm checkpoints from real PPO."""

    return "dream_episode_steps" in config and "wm_frameskip" in config


def _evaluate_snapshot(agent, settings: EvaluationSettings, snapshot_dir: Path):
    from src.evaluation.evaluate_pusht import make_repeat_seeds
    from src.evaluation.pusht import (
        PushTEvalConfig,
        aggregate_evaluation_results,
        run_evaluation,
    )

    results = []
    repeat_seeds = make_repeat_seeds(
        settings.seed,
        settings.repeats,
        settings.episodes,
        settings.seed_stride,
    )
    for repeat, repeat_seed in enumerate(repeat_seeds):
        video_dir = snapshot_dir / "videos" / f"repeat_{repeat:02d}_seed_{repeat_seed}"
        config = PushTEvalConfig(
            env_id=settings.env_id,
            episodes=settings.episodes,
            seed=repeat_seed,
            seed_stride=settings.seed_stride,
            max_episode_steps=settings.max_episode_steps,
            fixed_target_pose=settings.fixed_target_pose,
            fixed_target_block_success=settings.fixed_target_block_success,
            fixed_target_max_reset_attempts=settings.fixed_target_max_reset_attempts,
            agent_block_coef=settings.agent_block_coef,
            block_start_radius=settings.block_start_radius,
            record_video=settings.record_video,
            video_dir=str(video_dir),
            video_fps=settings.video_fps,
            video_resolution=settings.video_resolution,
            capture_traces=settings.capture_traces,
        )
        print(
            f"    repeat {repeat + 1}/{settings.repeats}: "
            f"seeds {repeat_seed}.."
            f"{repeat_seed + (settings.episodes - 1) * settings.seed_stride}"
        )
        results.append(run_evaluation(agent, config))
    return aggregate_evaluation_results(results)


def _write_metrics(path: Path, result, snapshot: dict, settings: EvaluationSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["snapshot"] = {
        "checkpoint": str(snapshot["path"].resolve()),
        "iteration": snapshot["iteration"],
        "filename_env_steps": snapshot["env_steps"],
    }
    payload["evaluation_id"] = settings.evaluation_id
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    snapshots = snapshot_checkpoints(run_dir)
    if not snapshots:
        raise SystemExit(
            f"no snapshot_step<env_steps>_it<iteration>.pt files found in {run_dir}"
        )

    from src.evaluation.agents import load_ppo_components, make_ppo_evaluation_agent

    # The first load supplies both the task configuration and the encoder reused
    # below. It also fails before any output is created if the checkpoint is bad.
    first_components = load_ppo_components(snapshots[0]["path"], args.device)
    if not _is_dream_config(first_components.config):
        raise SystemExit(
            f"{snapshots[0]['path']} is not a Dream-PPO checkpoint produced by "
            "src.ppo.train_lewm"
        )
    settings = resolve_settings(args, first_components.config)

    output_dir = (args.output_dir or run_dir / "snapshot_evaluations").expanduser().resolve()
    results_path = output_dir / "results.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []
    done = {
        (row.get("checkpoint"), row.get("evaluation_id"))
        for row in existing
        if not args.overwrite and Path(row.get("metrics_path", "")).is_file()
    }

    protocol = asdict(settings)
    protocol["evaluation_id"] = settings.evaluation_id
    print(f"Dream-PPO run: {run_dir}")
    print(f"Snapshots: {len(snapshots)}")
    print(
        f"Evaluation protocol {settings.evaluation_id}: "
        f"{json.dumps(protocol, sort_keys=True)}"
    )
    print(f"Results: {results_path}")

    encoder = first_components.encoder
    for index, snapshot in enumerate(snapshots, start=1):
        checkpoint = str(snapshot["path"].resolve())
        key = (checkpoint, settings.evaluation_id)
        label = f"it{snapshot['iteration']:05d}"
        if key in done:
            print(f"[{index}/{len(snapshots)}] {label}: already evaluated, skipping")
            continue

        print(f"[{index}/{len(snapshots)}] {label}: {snapshot['path'].name}")
        components = (
            first_components
            if index == 1
            else load_ppo_components(snapshot["path"], args.device, encoder=encoder)
        )
        if not _is_dream_config(components.config):
            raise RuntimeError(f"mixed non-Dream checkpoint in run: {snapshot['path']}")
        agent = make_ppo_evaluation_agent(
            checkpoint=snapshot["path"],
            components=components,
            deterministic=settings.deterministic,
            execution_mode=settings.execution_mode,
            replan_interval=settings.replan_interval,
            temporal_ensemble_decay=settings.temporal_ensemble_decay,
        )
        snapshot_dir = output_dir / settings.evaluation_id / snapshot["path"].stem
        result = _evaluate_snapshot(agent, settings, snapshot_dir)
        metrics_path = snapshot_dir / "metrics.json"
        _write_metrics(metrics_path, result, snapshot, settings)

        budget = checkpoint_budget(snapshot["path"])
        row = {
            "checkpoint": checkpoint,
            "iteration": snapshot["iteration"],
            "filename_env_steps": snapshot["env_steps"],
            "evaluation_id": settings.evaluation_id,
            "evaluation": protocol,
            "metrics_path": str(metrics_path),
            "repeat_success_rates": [
                repeat.summary["success_rate"] for repeat in result.results
            ],
            **result.summary,
            **budget,
        }
        upsert_jsonl(results_path, row, RESULT_KEY)
        print(
            f"    success={row['success_rate']:.3f} | "
            f"return={row['mean_return']:.3f} | episodes={row['episodes']}"
        )

    print(f"Completed snapshot evaluation. Results: {results_path}")


if __name__ == "__main__":
    main()
