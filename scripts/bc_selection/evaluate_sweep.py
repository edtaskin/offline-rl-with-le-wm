"""Evaluate the BC-checkpoint selection sweep: one row per (BC checkpoint, seed).

Every PPO number is produced by the canonical evaluator over 150 episodes sampled
reproducibly from one master seed, on the start-state distribution the agents
were trained on -- the same ``rq_common`` plumbing RQ1 reports through, so a
number here and a number on the poster mean the same thing.

Two things are recorded besides the headline success rate, because "we used
epoch 60" is a weak thing to defend and a mechanism is not:

* ``bc_standalone_success`` -- each BC checkpoint evaluated as a policy in its
  own right. If PPO peaks at a checkpoint whose standalone success is *not* the
  best, the sweep has found something: the better-fit BC is the worse RL
  initialization.
* ``bc_weight_l2`` / ``bc_last_layer_l2`` -- weight-norm growth over BC training,
  a cheap proxy for how much plasticity the initialization has left.

Seeds carry a role. ``--select-seeds`` are the seeds the choice is made on;
``--confirm-seeds`` are fresh seeds used to re-measure the winner afterwards.
The distinction matters because the selection seeds' mean is optimistically
biased by exactly the effect being claimed, so the report keeps them apart.

Usage::

    python -m scripts.bc_selection.evaluate_sweep --bc-dir runs/bc_long --select-seeds 1 2 3
    python -m scripts.bc_selection.evaluate_sweep \
        --bc-checkpoints runs/bc_long/pusht_bc_epoch060.pth --confirm-seeds 11 12 13
"""

from __future__ import annotations

import argparse
import re
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
    repo_path,
    run_canonical_eval,
    upsert_jsonl,
)

# Identifies one row in results.jsonl; a re-evaluation replaces its predecessor.
RESULT_KEY = ("bc_tag", "method", "variant", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bc-dir",
        default=None,
        help="Directory of BC checkpoints (*.pth, excluding the *_stats.pth sidecar)",
    )
    parser.add_argument(
        "--bc-checkpoints", nargs="+", default=None, help="Explicit BC checkpoint paths"
    )
    parser.add_argument(
        "--bc-stats",
        default=None,
        help="BC stats sidecar; defaults to the single *_stats.pth in --bc-dir",
    )
    parser.add_argument("--exp-prefix", default="bcsel_dream")
    parser.add_argument(
        "--select-seeds",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Seeds the BC choice is made on",
    )
    parser.add_argument(
        "--confirm-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Fresh seeds for re-measuring the winner; excluded from selection",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/bc_selection")
    parser.add_argument(
        "--variants", nargs="+", default=["best"], choices=["best", "final"]
    )
    parser.add_argument("--episodes", type=int, default=CANONICAL_EVAL["episodes"])
    parser.add_argument("--eval-seed", type=int, default=CANONICAL_EVAL["seed"])
    parser.add_argument(
        "--block-start-radius", type=float, default=CANONICAL_EVAL["block_start_radius"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-bc-standalone",
        action="store_true",
        help="Skip evaluating each BC checkpoint as a policy in its own right",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-evaluate rows already present in results.jsonl (default: skip them)",
    )
    return parser.parse_args()


def _bc_tag(checkpoint: str | Path) -> str:
    """Filename stem with non-alphanumerics folded to ``_``.

    Must stay identical to ``bc_tag()`` in ``run_sweep.sh``: it is the only link
    between a BC checkpoint and the ``exp_name`` its runs were trained under.
    """
    stem = Path(str(checkpoint)).name
    if stem.endswith(".pth"):
        stem = stem[: -len(".pth")]
    return re.sub(r"[^a-zA-Z0-9]", "_", stem).rstrip("_")


def _bc_epoch(tag: str) -> int | None:
    """Training epoch encoded by ``train_bc_latent.py``'s ``_epoch<N>`` suffix.

    ``None`` for the run's final checkpoint, which carries no suffix; the report
    places those after the largest numbered epoch.
    """
    match = re.search(r"epoch(\d+)", tag)
    return int(match.group(1)) if match else None


def discover_checkpoints(args) -> tuple[list[Path], Path]:
    if args.bc_checkpoints:
        checkpoints = [repo_path(path) for path in args.bc_checkpoints]
    elif args.bc_dir:
        directory = repo_path(args.bc_dir)
        # Numeric epoch order: train_bc_latent.py does not zero-pad the suffix,
        # so a plain lexicographic sort puts _epoch10 before _epoch2. The
        # unsuffixed final checkpoint sorts last.
        checkpoints = sorted(
            (path for path in directory.glob("*.pth") if not path.name.endswith("_stats.pth")),
            key=lambda path: (_bc_epoch(_bc_tag(path)) is None, _bc_epoch(_bc_tag(path)) or 0),
        )
    else:
        raise SystemExit("Pass --bc-dir or --bc-checkpoints.")
    if not checkpoints:
        raise SystemExit("No BC checkpoints found.")

    if args.bc_stats:
        stats = repo_path(args.bc_stats)
    else:
        directory = repo_path(args.bc_dir) if args.bc_dir else checkpoints[0].parent
        candidates = sorted(directory.glob("*_stats.pth"))
        if len(candidates) != 1:
            raise SystemExit(
                f"Expected exactly one *_stats.pth in {directory}, found {len(candidates)}. "
                "Pass --bc-stats explicitly."
            )
        stats = candidates[0]
    return checkpoints, stats


def bc_weight_diagnostics(checkpoint: str | Path) -> dict:
    """L2 norms of a BC checkpoint's weights.

    Weight norm grows monotonically over BC training and is the cheapest
    available stand-in for "how much plasticity is left in this initialization",
    which is the mechanism most likely to explain a non-monotone sweep.
    """
    import torch

    from src.utils.hf_hub import resolve_artifact

    state = torch.load(resolve_artifact(str(checkpoint)), map_location="cpu")
    total = 0.0
    weight_keys = [key for key in state if key.endswith(".weight")]
    for tensor in state.values():
        if torch.is_floating_point(tensor):
            total += float(tensor.pow(2).sum())
    last = state[weight_keys[-1]] if weight_keys else None
    return {
        "bc_weight_l2": total**0.5,
        "bc_last_layer_l2": float(last.norm()) if last is not None else None,
    }


def already_done(existing: list[dict], bc_tag, method, variant, seed) -> bool:
    return any(
        row.get("bc_tag") == bc_tag
        and row.get("method") == method
        and row.get("variant") == variant
        and row.get("seed") == seed
        for row in existing
    )


def evaluate_one(args, *, run_name, agent_type, checkpoint, stats=None, fields=None) -> dict:
    result = run_canonical_eval(
        agent_type,
        checkpoint,
        output_root=repo_path(args.output_root) / "evaluations",
        run_name=run_name,
        stats=stats,
        device=args.device,
        episodes=args.episodes,
        seed=args.eval_seed,
        block_start_radius=args.block_start_radius,
    )
    summary = result.summary
    row = {
        "agent_type": agent_type,
        "checkpoint": str(checkpoint),
        "success_rate": summary["success_rate"],
        "mean_length": summary["mean_length"],
        "mean_return": summary["mean_return"],
        "episodes": summary["episodes"],
        "eval": {
            "episodes": args.episodes,
            "seed": args.eval_seed,
            "episode_seed_sampling": "random-well-separated",
            "block_start_radius": args.block_start_radius,
        },
        "metrics_path": getattr(result, "metrics_path", None),
    }
    row.update(fields or {})
    return row


def main() -> None:
    args = parse_args()
    checkpoints, stats = discover_checkpoints(args)
    results_path = repo_path(args.output_root) / "results.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []

    overlap = set(args.select_seeds) & set(args.confirm_seeds)
    if overlap:
        raise SystemExit(
            f"Seeds {sorted(overlap)} are in both --select-seeds and --confirm-seeds. "
            "A confirmation seed must not be one the choice was made on."
        )
    seed_roles = [(seed, "select") for seed in args.select_seeds]
    seed_roles += [(seed, "confirm") for seed in args.confirm_seeds]

    print(f"{len(checkpoints)} BC checkpoints | stats: {stats}")
    for checkpoint in checkpoints:
        tag = _bc_tag(checkpoint)
        epoch = _bc_epoch(tag)
        diagnostics = bc_weight_diagnostics(checkpoint)
        common = {"bc_tag": tag, "bc_epoch": epoch, "bc_checkpoint": str(checkpoint), **diagnostics}

        # The BC as a policy in its own right -- one row per checkpoint, no seed:
        # BC training is deterministic given its own seed, and this sweep varies
        # the PPO seed only.
        if not args.no_bc_standalone and (
            args.overwrite or not already_done(existing, tag, "bc", "standalone", None)
        ):
            row = evaluate_one(
                args,
                run_name=f"bc_{tag}",
                agent_type="bc",
                checkpoint=checkpoint,
                stats=stats,
                fields={**common, "method": "bc", "variant": "standalone", "seed": None,
                        "seed_role": None},
            )
            upsert_jsonl(results_path, row, RESULT_KEY)
            existing.append(row)
            print(f"[bc/{tag}] standalone success={row['success_rate']:.3f}")

        exp_name = f"{args.exp_prefix}_{tag}"
        for seed, role in seed_roles:
            try:
                run = find_run_dir(exp_name, seed, args.runs_root)
            except FileNotFoundError as exc:
                print(f"[dream_ppo/{tag}/seed{seed}] SKIPPED: {exc}")
                continue

            for variant in args.variants:
                if already_done(existing, tag, "dream_ppo", variant, seed) and not args.overwrite:
                    print(f"[dream_ppo/{tag}/{variant}/seed{seed}] already recorded, skipping")
                    continue
                checkpoint_path = run.path / f"{variant}.pt"
                if not checkpoint_path.exists():
                    print(f"[dream_ppo/{tag}/{variant}/seed{seed}] SKIPPED: {checkpoint_path} missing")
                    continue

                budget = checkpoint_budget(checkpoint_path)
                row = evaluate_one(
                    args,
                    run_name=f"dream_{tag}_{variant}_seed{seed}",
                    agent_type="ppo",
                    checkpoint=checkpoint_path,
                    fields={
                        **common,
                        **budget,
                        "method": "dream_ppo",
                        "variant": variant,
                        "seed": seed,
                        "seed_role": role,
                        "run_dir": str(run.path),
                    },
                )
                upsert_jsonl(results_path, row, RESULT_KEY)
                existing.append(row)
                print(
                    f"[dream_ppo/{tag}/{variant}/seed{seed}] ({role}) "
                    f"success={row['success_rate']:.3f}"
                )

    print(f"\nresults -> {results_path}")
    print("Next: python -m scripts.bc_selection.report")


if __name__ == "__main__":
    main()
