"""Evaluate every RQ1 checkpoint under one fixed protocol, resumably.

RQ1 asks whether policy improvement can happen entirely inside the world model.
Answering it means putting BC, real-env PPO and dream PPO on exactly the same
evaluator (``src.evaluation.evaluate_pusht``) and recording, next to each
success rate, the number of real ``env.step`` calls that checkpoint cost --
including the ones spent selecting it.

Each checkpoint is evaluated through the canonical entry point, so results are
identical to running that module by hand; this script only adds the grid, the
interaction-budget bookkeeping, and a ``results.jsonl`` ledger it can resume
from (a full grid is hours of simulation, and interrupting it is normal).

Examples::

    # headline table: one row per agent x training seed
    python -m scripts.rq1.evaluate_grid \
        --bc hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth \
        --run real-ppo=runs/latent_ppo_pusht_sparse_circle__seed1/14072026-202921 \
        --run dream-ppo=runs/latent_ppo_pusht_lewm_dream__seed1/16072026-144748 \
        --checkpoints best final

    # interaction-budget curve: every step-tagged snapshot of a real-PPO run
    python -m scripts.rq1.evaluate_grid \
        --run real-ppo=runs/latent_ppo_pusht_sparse_circle__seed2/<stamp> \
        --checkpoints snapshots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from src.evaluation.evaluate_pusht import build_parser as build_eval_parser
from src.evaluation.evaluate_pusht import evaluate_from_args


DEFAULT_LEDGER = "runs/rq1/results.jsonl"
DEFAULT_BC_STATS = "hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="a training run directory to evaluate, e.g. dream-ppo=runs/<exp>__seed1/<stamp>",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best", "final"],
        help="checkpoint names within each run dir; 'snapshots' expands to every "
        "snapshot_step*.pt (the interaction-budget curve)",
    )
    parser.add_argument("--bc", default=None, help="BC checkpoint to evaluate as the zero-interaction baseline")
    parser.add_argument("--bc-stats", default=DEFAULT_BC_STATS)
    parser.add_argument("--bc-label", default="bc")
    parser.add_argument("--bc-seed", type=int, default=None, help="training seed to record for the BC row")

    # Evaluation protocol. Defaults are the README's canonical settings; the
    # block-start radius matches the PPO/dream training start distribution.
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="gap between episode seeds; use >=7 for unrestricted starts",
    )
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--block-start-radius", type=float, default=200.0)
    parser.add_argument(
        "--no-block-start-radius",
        dest="block_start_radius",
        action="store_const",
        const=None,
        help="unrestricted block starts -- the whole workspace, not a disk around the goal",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="runs/evaluations")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--force", action="store_true", help="re-evaluate rows already in the ledger")
    parser.add_argument("--dry-run", action="store_true", help="list what would be evaluated, then exit")
    return parser


# --------------------------------------------------------------------- specs
def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--run expects LABEL=RUN_DIR, got {spec!r}")
    label, run_dir = spec.split("=", 1)
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    return label.strip(), path


def seed_from_run_dir(run_dir: Path) -> int | None:
    """Recover the training seed from the ``<exp>__seed<k>/<stamp>`` layout."""
    for part in run_dir.parts[::-1]:
        match = re.search(r"__seed(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def expand_checkpoints(run_dir: Path, names: list[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        if name == "snapshots":
            paths.extend(sorted(run_dir.glob("snapshot_step*.pt")))
            continue
        candidate = run_dir / (name if name.endswith(".pt") else f"{name}.pt")
        if candidate.exists():
            paths.append(candidate)
        else:
            print(f"  ! missing {candidate}, skipping")
    return paths


def checkpoint_budget(path: Path) -> dict:
    """Interaction budget recorded in a checkpoint, tolerating older files.

    Checkpoints written before env-step accounting existed only carry
    ``global_step``. For a real-env run that is the training interaction, so it
    is a sound fallback; for a dream run it counts *imagined* steps and cannot be
    reinterpreted, so it is reported as unknown rather than guessed at.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    is_dream = "wm_frameskip" in config
    if "env_steps_consumed" in payload:
        return {
            "env_steps_consumed": int(payload["env_steps_consumed"]),
            "train_env_steps": int(payload.get("train_env_steps", 0)),
            "eval_env_steps": int(payload.get("eval_env_steps", 0)),
            "budget_source": "recorded",
            "imagined_steps": int(payload.get("global_step", 0)),
            "selection": config.get("selection"),
            "train_success_rate": payload.get("success_rate"),
        }
    global_step = int(payload.get("global_step", 0))
    return {
        "env_steps_consumed": None if is_dream else global_step,
        "train_env_steps": None if is_dream else global_step,
        "eval_env_steps": None,
        "budget_source": "legacy_dream_global_step" if is_dream else "legacy_global_step",
        "imagined_steps": global_step,
        "selection": config.get("selection"),
        "train_success_rate": payload.get("success_rate"),
    }


# -------------------------------------------------------------------- ledger
def protocol_key(args) -> str:
    stride = "" if args.seed_stride == 1 else f"_stride{args.seed_stride}"
    return (
        f"eps{args.episodes}_rep{args.repeats}_seed{args.seed}{stride}"
        f"_max{args.max_episode_steps}_radius{args.block_start_radius}"
    )


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


# ----------------------------------------------------------------- evaluation
def evaluate_one(args, agent_type: str, checkpoint: str, run_name: str, stats: str | None = None):
    """Run the canonical evaluator; returns its pooled result."""
    argv = [
        "--agent-type", agent_type,
        "--checkpoint", str(checkpoint),
        "--episodes", str(args.episodes),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--seed-stride", str(args.seed_stride),
        "--max-episode-steps", str(args.max_episode_steps),
        "--device", args.device,
        "--output-root", args.output_root,
        "--run-name", run_name,
    ]
    if args.block_start_radius is not None:
        argv += ["--block-start-radius", str(args.block_start_radius)]
    if stats:
        argv += ["--stats", stats]
    return evaluate_from_args(build_eval_parser().parse_args(argv))


def main() -> None:
    args = build_parser().parse_args()
    ledger_path = Path(args.ledger)
    protocol = protocol_key(args)
    done = {
        (row["label"], row["checkpoint"], row["protocol"])
        for row in load_ledger(ledger_path)
    }

    jobs: list[dict] = []
    if args.bc:
        jobs.append(
            {
                "label": args.bc_label,
                "agent_type": "bc",
                "checkpoint": args.bc,
                "stats": args.bc_stats,
                "seed": args.bc_seed,
                "checkpoint_name": "bc",
                # BC is trained purely offline: the zero-interaction reference
                # point the whole RQ is measured against.
                "budget": {
                    "env_steps_consumed": 0,
                    "train_env_steps": 0,
                    "eval_env_steps": 0,
                    "budget_source": "offline_bc",
                    "imagined_steps": 0,
                    "selection": None,
                    "train_success_rate": None,
                },
            }
        )

    for spec in args.run:
        label, run_dir = parse_run_spec(spec)
        seed = seed_from_run_dir(run_dir)
        for checkpoint in expand_checkpoints(run_dir, args.checkpoints):
            jobs.append(
                {
                    "label": label,
                    "agent_type": "ppo",
                    "checkpoint": str(checkpoint),
                    "stats": None,
                    "seed": seed,
                    "checkpoint_name": checkpoint.stem,
                    "budget": checkpoint_budget(checkpoint),
                }
            )

    pending = [job for job in jobs if args.force or (job["label"], job["checkpoint"], protocol) not in done]
    print(f"RQ1 grid | protocol {protocol} | {len(jobs)} checkpoints, {len(pending)} pending")
    for job in pending:
        print(f"  - {job['label']:12s} seed={job['seed']} {job['checkpoint']}")
    if args.dry_run or not pending:
        return

    episodes_per_eval = args.episodes * args.repeats
    for index, job in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {job['label']} :: {job['checkpoint']}")
        run_name = f"rq1_{job['label']}_seed{job['seed']}_{job['checkpoint_name']}"
        result = evaluate_one(
            args,
            job["agent_type"],
            job["checkpoint"],
            run_name,
            stats=job["stats"],
        )
        summary = result.summary
        row = {
            "label": job["label"],
            "agent_type": job["agent_type"],
            "seed": job["seed"],
            "checkpoint": job["checkpoint"],
            "checkpoint_name": job["checkpoint_name"],
            "protocol": protocol,
            "episodes": episodes_per_eval,
            "success_rate": summary["success_rate"],
            "mean_length": summary["mean_length"],
            "mean_return": summary["mean_return"],
            "repeat_success_rates": [r.summary["success_rate"] for r in result.results],
            "repeat_seeds": list(result.repeat_seeds),
            **job["budget"],
        }
        append_ledger(ledger_path, row)
        print(f"  -> success {summary['success_rate']:.3f} over {episodes_per_eval} episodes; logged to {ledger_path}")


if __name__ == "__main__":
    main()
