"""RQ1 headline table: BC vs real-env PPO vs dream PPO, over training seeds.

Every number is produced by the canonical evaluator,
``python -m src.evaluation.evaluate_pusht --repeats 3 --episodes 50``, on the
start-state distribution the agents were trained on
(``--block-start-radius 200``). Three repeats with non-overlapping seed ranges
give 150 episodes per checkpoint; the spread across *training* seeds is reported
separately by ``scripts/rq1/report.py``.

The point of RQ1 is not only "how good", it is "bought with how much environment
interaction", so each row also carries the interaction budget recorded inside the
checkpoint: rollout steps plus the steps spent on real-env checkpoint selection.

Three checkpoint variants per PPO run make the selection story explicit:

* ``final``          -- the last checkpoint. No selection at all, so its budget is
                        purely training. For dream PPO this is the honest,
                        genuinely interaction-free number.
* ``best``           -- ``best.pt``, selected by whatever ``--selection`` the run
                        used (``dream`` for the dream runs = zero env steps,
                        held-out real success for the real runs = extra env steps).
* ``best_real_sel``  -- (dream runs only) the snapshot that the *real* held-out
                        metric would have picked, reconstructed from
                        ``selection_log.jsonl``. This is the contrast that shows
                        what interaction-free selection gives up -- and it is not
                        an interaction-free number.

Usage::

    python -m scripts.rq1.evaluate_grid --seeds 1 2 3
    python -m scripts.rq1.evaluate_grid --seeds 1 2 3 --variants final --no-bc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rq_common import (
    CANONICAL_EVAL,
    checkpoint_budget,
    find_run_dir,
    read_jsonl,
    read_selection_log,
    repo_path,
    run_canonical_eval,
    snapshot_checkpoints,
    upsert_jsonl,
)

BC_CHECKPOINT = "hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth"
BC_STATS = "hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth"

# Identifies one row in results.jsonl; a re-evaluation replaces its predecessor.
RESULT_KEY = ("method", "variant", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-exp", default="latent_ppo_pusht_real_dense_bc_v2")
    parser.add_argument("--dream-exp", default="latent_ppo_pusht_lewm_sparse_dense_correlation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/rq1")
    parser.add_argument("--episodes", type=int, default=CANONICAL_EVAL["episodes"])
    parser.add_argument("--repeats", type=int, default=CANONICAL_EVAL["repeats"])
    parser.add_argument("--eval-seed", type=int, default=CANONICAL_EVAL["seed"])
    parser.add_argument(
        "--block-start-radius", type=float, default=CANONICAL_EVAL["block_start_radius"]
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["final", "best", "best_real_sel"],
        choices=["final", "best", "best_real_sel"],
    )
    parser.add_argument("--bc-checkpoint", default=BC_CHECKPOINT)
    parser.add_argument("--bc-stats", default=BC_STATS)
    parser.add_argument("--no-bc", action="store_true", help="skip the BC baseline row")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-evaluate rows already present in results.jsonl (default: skip them)",
    )
    return parser.parse_args()


def real_selected_snapshot(run_dir: Path):
    """The snapshot the *real* held-out metric would have selected.

    Ties go to the earliest iteration: if two checkpoints score the same, the
    cheaper one is what a real selection rule running online would have kept.
    """
    rows = [row for row in read_selection_log(run_dir) if row.get("real_success") is not None]
    if not rows:
        return None, None
    best = max(rows, key=lambda row: (row["real_success"], -row["iteration"]))
    for snapshot in snapshot_checkpoints(run_dir):
        if snapshot["iteration"] == best["iteration"]:
            return snapshot, best
    return None, best


def already_done(existing: list[dict], method: str, variant: str, seed) -> bool:
    return any(
        row["method"] == method and row["variant"] == variant and row["seed"] == seed
        for row in existing
    )


def evaluate_one(args, *, method, variant, seed, agent_type, checkpoint, stats=None, extra_fields=None):
    run_name = f"{method}_{variant}_seed{seed}" if seed is not None else f"{method}_{variant}"
    result = run_canonical_eval(
        agent_type,
        checkpoint,
        output_root=Path(args.output_root) / "evaluations",
        run_name=run_name,
        stats=stats,
        device=args.device,
        episodes=args.episodes,
        repeats=args.repeats,
        seed=args.eval_seed,
        block_start_radius=args.block_start_radius,
    )
    summary = result.summary
    row = {
        "method": method,
        "variant": variant,
        "seed": seed,
        "agent_type": agent_type,
        "checkpoint": str(checkpoint),
        "success_rate": summary["success_rate"],
        "mean_length": summary["mean_length"],
        "mean_return": summary["mean_return"],
        "episodes": summary["episodes"],
        "repeats": summary["repeats"],
        "repeat_success_rates": [r.summary["success_rate"] for r in result.results],
        "eval": {
            "episodes_per_repeat": args.episodes,
            "repeats": args.repeats,
            "seed": args.eval_seed,
            "block_start_radius": args.block_start_radius,
        },
        "metrics_path": getattr(result, "metrics_path", None),
    }
    row.update(extra_fields or {})
    return row


def main() -> None:
    args = parse_args()
    results_path = repo_path(args.output_root) / "results.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []

    if not args.no_bc and (args.overwrite or not already_done(existing, "bc", "published", None)):
        # One published BC checkpoint backs every PPO seed, so BC is a single row.
        # The consequence -- seed spread covers PPO/dream RNG only, not the BC
        # prior -- is carried into the report as a stated caveat.
        row = evaluate_one(
            args,
            method="bc",
            variant="published",
            seed=None,
            agent_type="bc",
            checkpoint=args.bc_checkpoint,
            stats=args.bc_stats,
            extra_fields={
                "env_steps_consumed": 0,
                "train_env_steps": 0,
                "eval_env_steps": 0,
                "required_env_steps": 0,
                "diagnostic_env_steps": 0,
                "interaction_free": True,
                "selection": "none",
            },
        )
        upsert_jsonl(results_path, row, RESULT_KEY)
        print(f"[bc/published] success={row['success_rate']:.3f}")

    for method, exp_name in (("real_ppo", args.real_exp), ("dream_ppo", args.dream_exp)):
        for seed in args.seeds:
            try:
                run = find_run_dir(exp_name, seed, args.runs_root)
            except FileNotFoundError as exc:
                print(f"[{method}/seed{seed}] SKIPPED: {exc}")
                continue
            print(f"[{method}/seed{seed}] run dir: {run.path}")

            for variant in args.variants:
                if variant == "best_real_sel" and method != "dream_ppo":
                    continue  # real-PPO's best.pt already *is* the real-selected one
                if already_done(existing, method, variant, seed) and not args.overwrite:
                    print(f"[{method}/{variant}/seed{seed}] already in results.jsonl, skipping")
                    continue

                selection_note = None
                if variant == "best_real_sel":
                    snapshot, log_row = real_selected_snapshot(run.path)
                    if snapshot is None:
                        print(
                            f"[{method}/{variant}/seed{seed}] SKIPPED: no snapshot matches the "
                            "best real_success row (train with --snapshot-interval matching "
                            "--eval-interval and --record-real-eval)"
                        )
                        continue
                    checkpoint = snapshot["path"]
                    selection_note = {
                        "selected_iteration": log_row["iteration"],
                        "selected_on_real_success": log_row["real_success"],
                    }
                else:
                    checkpoint = run.path / f"{variant}.pt"
                    if not checkpoint.exists():
                        print(f"[{method}/{variant}/seed{seed}] SKIPPED: {checkpoint} missing")
                        continue

                budget = checkpoint_budget(checkpoint)
                # Interaction-free means: no env step was spent either training
                # this policy or deciding to keep it. Dream training spends none;
                # a real-env selection rule does, which is exactly the cost RQ1
                # is trying to expose.
                if method == "dream_ppo":
                    selects_on_real = variant == "best_real_sel" or budget.get("selection") == "real"
                    interaction_free = not selects_on_real
                    if variant == "best" and budget.get("selection") != "dream":
                        print(
                            f"[{method}/best/seed{seed}] WARNING: run used "
                            f"selection={budget.get('selection')!r}, so best.pt is NOT "
                            "interaction-free. Re-train with --selection dream."
                        )
                else:
                    interaction_free = False

                # What this checkpoint *needed*, as opposed to what its run
                # happened to spend. A dream run with --selection dream needs
                # zero: its real-env eval is RQ2 instrumentation that a real
                # deployment would simply not run. Anything selected on the
                # simulator needs the selection steps too, which is the cost RQ1
                # exists to expose -- so it is never discounted away.
                required = (
                    budget["train_env_steps"] if interaction_free else budget["env_steps_consumed"]
                )
                row = evaluate_one(
                    args,
                    method=method,
                    variant=variant,
                    seed=seed,
                    agent_type="ppo",
                    checkpoint=checkpoint,
                    extra_fields={
                        **budget,
                        "required_env_steps": required,
                        "diagnostic_env_steps": budget["env_steps_consumed"] - required,
                        "run_dir": str(run.path),
                        "interaction_free": interaction_free,
                        **(selection_note or {}),
                    },
                )
                upsert_jsonl(results_path, row, RESULT_KEY)
                existing.append(row)
                diagnostic = row["diagnostic_env_steps"]
                note = f", {diagnostic:,} diagnostic-only" if diagnostic else ""
                print(
                    f"[{method}/{variant}/seed{seed}] success={row['success_rate']:.3f} "
                    f"| env steps required={row['required_env_steps']:,} "
                    f"(train {row['train_env_steps']:,} + selection {row['eval_env_steps']:,}{note})"
                )

    print(f"\nresults -> {results_path}")
    print("Next: python -m scripts.rq1.report")


if __name__ == "__main__":
    main()
