"""RQ1 interaction-budget curve: success vs environment steps consumed.

Evaluates the step-tagged snapshots a training run leaves behind
(``snapshot_step<env_steps>_it<iteration>.pt``) so real-env PPO can be placed on
an x-axis of *environment interaction*, which is the axis dream PPO sits at zero
on. The budget on that axis is what the checkpoint actually cost: rollout steps
plus any steps spent evaluating in the simulator to pick it.

Snapshots are evaluated with the same loop as the headline table -- the same
:class:`PushTEvalConfig`, the same non-overlapping repeat seeds, a fresh env per
repeat -- but with one frozen ViT amortized across every snapshot in a run, since
rebuilding the encoder dozens of times would dominate the runtime. Curve points
default to fewer episodes than the headline table (one repeat instead of three):
a curve needs many cheap points, the table needs few precise ones.

The dream-PPO reference level is *not* computed here; it comes from
``results.jsonl`` and is drawn by ``scripts/rq1/report.py`` as a horizontal line
at x = 0.

Usage::

    python -m scripts.rq1.budget_curve --seeds 1 2 3
    python -m scripts.rq1.budget_curve --seeds 1 2 3 --every 1 --repeats 2
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
    evaluate_agent_repeats,
    find_run_dir,
    read_jsonl,
    repo_path,
    snapshot_checkpoints,
    upsert_jsonl,
)

# Identifies one curve point; a re-evaluation replaces its predecessor.
CURVE_KEY = ("method", "seed", "iteration")

EXP_NAMES = {
    "real_ppo": "latent_ppo_pusht_real_dense_bc_v2",
    "dream_ppo": "latent_ppo_pusht_lewm_sparse_dense_correlation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", default=["real_ppo"], choices=sorted(EXP_NAMES))
    parser.add_argument("--real-exp", default=EXP_NAMES["real_ppo"])
    parser.add_argument("--dream-exp", default=EXP_NAMES["dream_ppo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/rq1")
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="evaluate every Nth snapshot (the first and last are always kept)",
    )
    parser.add_argument("--episodes", type=int, default=CANONICAL_EVAL["episodes"])
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeats per curve point (default 1; the headline table uses 3)",
    )
    parser.add_argument("--eval-seed", type=int, default=CANONICAL_EVAL["seed"])
    parser.add_argument(
        "--block-start-radius", type=float, default=CANONICAL_EVAL["block_start_radius"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_snapshots(snapshots: list[dict], every: int) -> list[dict]:
    """Subsample while keeping the endpoints: the BC prior and the final policy."""
    if every <= 1 or len(snapshots) <= 2:
        return snapshots
    kept = {0, len(snapshots) - 1}
    kept.update(range(0, len(snapshots), every))
    return [snapshots[index] for index in sorted(kept)]


def main() -> None:
    args = parse_args()
    from src.evaluation.agents import load_ppo_components, make_ppo_evaluation_agent

    curve_path = repo_path(args.output_root) / "budget_curve.jsonl"
    existing = read_jsonl(curve_path) if curve_path.exists() else []
    done = {
        (row["method"], row["seed"], row["iteration"])
        for row in existing
        if not args.overwrite
    }

    exp_names = {"real_ppo": args.real_exp, "dream_ppo": args.dream_exp}
    for method in args.methods:
        for seed in args.seeds:
            try:
                run = find_run_dir(exp_names[method], seed, args.runs_root)
            except FileNotFoundError as exc:
                print(f"[{method}/seed{seed}] SKIPPED: {exc}")
                continue

            snapshots = snapshot_checkpoints(run.path)
            if not snapshots:
                print(
                    f"[{method}/seed{seed}] SKIPPED: no snapshots in {run.path}. "
                    "Re-train with --snapshot_interval 10 (see scripts/rq1/run_campaign.sh)."
                )
                continue
            selected = select_snapshots(snapshots, args.every)
            print(
                f"[{method}/seed{seed}] {len(selected)}/{len(snapshots)} snapshots "
                f"from {run.path}"
            )

            encoder = None
            for snapshot in selected:
                key = (method, seed, snapshot["iteration"])
                if key in done:
                    print(f"  it{snapshot['iteration']:05d}: already evaluated, skipping")
                    continue

                components = load_ppo_components(snapshot["path"], args.device, encoder=encoder)
                encoder = components.encoder  # amortize the frozen ViT across the run
                agent = make_ppo_evaluation_agent(
                    components=components, deterministic=True, execution_mode="open-loop"
                )
                result = evaluate_agent_repeats(
                    agent,
                    episodes=args.episodes,
                    repeats=args.repeats,
                    seed=args.eval_seed,
                    block_start_radius=args.block_start_radius,
                )
                budget = checkpoint_budget(snapshot["path"])
                row = {
                    "method": method,
                    "seed": seed,
                    "iteration": snapshot["iteration"],
                    "checkpoint": str(snapshot["path"]),
                    "success_rate": result.summary["success_rate"],
                    "mean_length": result.summary["mean_length"],
                    "episodes": result.summary["episodes"],
                    **budget,
                }
                upsert_jsonl(curve_path, row, CURVE_KEY)
                print(
                    f"  it{snapshot['iteration']:05d} | env steps {row['env_steps_consumed']:>8d} "
                    f"| success {row['success_rate']:.3f} ({row['episodes']} eps)"
                )

    print(f"\nbudget curve -> {curve_path}")
    print("Next: python -m scripts.rq1.report")


if __name__ == "__main__":
    main()
