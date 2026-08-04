"""RQ2, part 1: how optimistic is the imagined reward, over a full training run?

The dream trainer already logs both numbers at the same cadence, for the same
weights, into ``selection_log.jsonl``:

* ``dream_success`` -- deterministic imagined rollouts from held-out expert
  anchors, scored by the frozen ``objective_met`` probe. This is what PPO is
  effectively maximizing.
* ``real_success``  -- the same weights evaluated in the actual simulator
  (``--record-real-eval``). Diagnostic only: with ``--selection dream`` it never
  influences which checkpoint is kept.

Their difference on a shared x-axis is the optimism. Two things matter for the
poster and both are computed here: the *level* gap (imagined success sits above
real success by how much) and whether the imagined metric at least **ranks**
checkpoints correctly, which is the weaker property interaction-free selection
would need to be usable at all.

Usage::

    python -m scripts.rq2.optimism_gap --seeds 1 2 3
    python -m scripts.rq2.optimism_gap --seeds 1 2 3 --x iteration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.rq_common import (
    INK,
    SERIES_COLORS,
    SERIES_LABELS,
    apply_figure_style,
    find_run_dir,
    pearson,
    read_selection_log,
    repo_path,
    save_figure,
    spearman,
)

X_AXES = {
    "imagined_steps": "imagined env-step equivalents",
    "iteration": "PPO iteration",
    "env_steps_consumed": "real environment steps consumed (diagnostic eval only)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dream-exp", default="latent_ppo_pusht_lewm_sparse_dense_correlation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/rq2")
    parser.add_argument("--x", default="imagined_steps", choices=sorted(X_AXES))
    return parser.parse_args()


def load_seed_curves(args) -> dict[int, dict[str, np.ndarray]]:
    curves = {}
    for seed in args.seeds:
        try:
            run = find_run_dir(args.dream_exp, seed, args.runs_root)
            rows = read_selection_log(run.path)
        except FileNotFoundError as exc:
            print(f"[seed {seed}] SKIPPED: {exc}")
            continue
        paired = [
            row
            for row in rows
            if row.get("dream_success") is not None and row.get("real_success") is not None
        ]
        if not paired:
            print(
                f"[seed {seed}] SKIPPED: no rows with both metrics. Train with "
                "--record-real-eval so dream and real are measured at the same weights."
            )
            continue
        curves[seed] = {
            "x": np.array([row[args.x] for row in paired], dtype=float),
            "dream": np.array([row["dream_success"] for row in paired], dtype=float),
            "real": np.array([row["real_success"] for row in paired], dtype=float),
            "run_dir": str(run.path),
            "dream_length_env_steps": np.array(
                [row.get("dream_length_env_steps") or np.nan for row in paired], dtype=float
            ),
        }
        print(f"[seed {seed}] {len(paired)} paired measurements from {run.path}")
    return curves


def summarize(curves: dict[int, dict]) -> dict:
    per_seed = {}
    all_dream, all_real = [], []
    for seed, curve in curves.items():
        dream, real = curve["dream"], curve["real"]
        gap = dream - real
        per_seed[seed] = {
            "measurements": int(len(dream)),
            "mean_dream_success": float(dream.mean()),
            "mean_real_success": float(real.mean()),
            "mean_gap": float(gap.mean()),
            "final_gap": float(gap[-1]),
            "max_gap": float(gap.max()),
            "dream_range": [float(dream.min()), float(dream.max())],
            "real_range": [float(real.min()), float(real.max())],
            "pearson_r": pearson(dream, real),
            "spearman_rho": spearman(dream, real),
            # What interaction-free selection would have cost: the real success
            # of the checkpoint the imagined metric likes best, against the real
            # success of the checkpoint a simulator-based rule would have kept.
            "real_success_at_dream_argmax": float(real[int(np.argmax(dream))]),
            "best_real_success": float(real.max()),
        }
        per_seed[seed]["selection_regret"] = (
            per_seed[seed]["best_real_success"]
            - per_seed[seed]["real_success_at_dream_argmax"]
        )
        all_dream.append(dream)
        all_real.append(real)

    pooled_dream = np.concatenate(all_dream)
    pooled_real = np.concatenate(all_real)
    return {
        "per_seed": per_seed,
        "pooled": {
            "measurements": int(len(pooled_dream)),
            "mean_gap": float((pooled_dream - pooled_real).mean()),
            "mean_dream_success": float(pooled_dream.mean()),
            "mean_real_success": float(pooled_real.mean()),
            "pearson_r": pearson(pooled_dream, pooled_real),
            "spearman_rho": spearman(pooled_dream, pooled_real),
            "mean_selection_regret": float(
                np.mean([stats["selection_regret"] for stats in per_seed.values()])
            ),
        },
    }


def plot_gap(curves, summary, args, output_root: Path) -> None:
    fig, (ax, ax_gap) = plt.subplots(
        2, 1, figsize=(7.0, 5.4), sharex=True, height_ratios=[2.4, 1.0]
    )

    for seed, curve in sorted(curves.items()):
        ax.plot(curve["x"], curve["dream"], color=SERIES_COLORS["dream"], alpha=0.28, linewidth=1.1)
        ax.plot(curve["x"], curve["real"], color=SERIES_COLORS["real"], alpha=0.28, linewidth=1.1)
        ax_gap.plot(
            curve["x"],
            curve["dream"] - curve["real"],
            color=INK["secondary"],
            alpha=0.25,
            linewidth=1.1,
        )

    # Seeds are logged at the same cadence, so a common grid is just their x.
    grid = np.unique(np.concatenate([curve["x"] for curve in curves.values()]))
    mean_dream = np.mean(
        [np.interp(grid, curve["x"], curve["dream"]) for curve in curves.values()], axis=0
    )
    mean_real = np.mean(
        [np.interp(grid, curve["x"], curve["real"]) for curve in curves.values()], axis=0
    )
    ax.plot(grid, mean_dream, color=SERIES_COLORS["dream"], linewidth=2.4, label=SERIES_LABELS["dream"])
    ax.plot(grid, mean_real, color=SERIES_COLORS["real"], linewidth=2.4, label=SERIES_LABELS["real"])
    ax.fill_between(grid, mean_real, mean_dream, color=SERIES_COLORS["dream"], alpha=0.10, linewidth=0)

    ax.annotate(
        SERIES_LABELS["dream"],
        xy=(grid[-1], mean_dream[-1]),
        xytext=(6, 2),
        textcoords="offset points",
        color=SERIES_COLORS["dream"],
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        SERIES_LABELS["real"],
        xy=(grid[-1], mean_real[-1]),
        xytext=(6, -8),
        textcoords="offset points",
        color=SERIES_COLORS["real"],
        fontsize=9,
        fontweight="bold",
    )

    pooled = summary["pooled"]
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "The imagined reward is optimistic: what PPO sees vs what the simulator says",
        pad=12,
    )
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.05,
        f"mean gap {pooled['mean_gap']:+.3f}   ·   Spearman ρ = {pooled['spearman_rho']:.2f}",
        transform=ax.transAxes,
        color=INK["secondary"],
        fontsize=9,
    )

    mean_gap = np.mean(
        [
            np.interp(grid, curve["x"], curve["dream"] - curve["real"])
            for curve in curves.values()
        ],
        axis=0,
    )
    ax_gap.plot(grid, mean_gap, color=INK["primary"], linewidth=2.0)
    ax_gap.axhline(0.0, color=INK["muted"], linewidth=1.0)
    ax_gap.set_ylabel("optimism\n(imagined − real)")
    ax_gap.set_xlabel(X_AXES[args.x])

    caption = "both metrics measured at the same weights, at the same cadence"
    if len(curves) > 1:
        caption = "thin lines = individual training seeds · " + caption
    fig.text(0.5, -0.03, caption, ha="center", color=INK["muted"], fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_root / "rq2_optimism_gap.png")
    plt.close(fig)


def plot_rank_scatter(curves, summary, output_root: Path) -> None:
    """Does the imagined metric at least rank checkpoints the way reality does?"""
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for seed, curve in sorted(curves.items()):
        ax.plot(
            curve["real"],
            curve["dream"],
            "o",
            markersize=6,
            color=SERIES_COLORS["dream"],
            markeredgecolor=INK["surface"],
            markeredgewidth=1.2,  # 2px surface ring on overlapping marks
            alpha=0.75,
        )
    # Zoom to the region the measurements occupy: the story is how far the cloud
    # sits above the diagonal, and empty axis space just shrinks that signal.
    observed = np.concatenate(
        [curve[key] for curve in curves.values() for key in ("dream", "real")]
    )
    lower = max(0.0, float(np.floor((observed.min() - 0.05) * 20) / 20))
    limits = [lower, 1.0]
    ax.plot(limits, limits, color=INK["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        "perfect agreement",
        xy=(limits[0] + 0.22 * (limits[1] - limits[0]),) * 2,
        xytext=(4, 4),
        textcoords="offset points",
        color=INK["muted"],
        fontsize=8,
        rotation=45,
        rotation_mode="anchor",
    )
    pooled = summary["pooled"]
    ax.set_xlabel("real held-out success")
    ax.set_ylabel("imagined (dream) success")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.set_title(f"Spearman ρ = {pooled['spearman_rho']:.2f}", pad=10)
    save_figure(fig, output_root / "rq2_dream_vs_real_scatter.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_figure_style()
    output_root = repo_path(args.output_root)

    curves = load_seed_curves(args)
    if not curves:
        raise SystemExit(
            "No usable selection logs. Train dream PPO with --record-real-eval "
            "(see scripts/rq1/run_campaign.sh)."
        )

    summary = summarize(curves)
    plot_gap(curves, summary, args, output_root / "figures")
    plot_rank_scatter(curves, summary, output_root / "figures")

    summary["runs"] = {seed: curve["run_dir"] for seed, curve in curves.items()}
    summary["x_axis"] = args.x
    path = output_root / "optimism_gap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")

    pooled = summary["pooled"]
    print(
        f"\nPooled over {len(curves)} seeds ({pooled['measurements']} paired measurements):\n"
        f"  imagined success {pooled['mean_dream_success']:.3f} vs real "
        f"{pooled['mean_real_success']:.3f}  ->  optimism {pooled['mean_gap']:+.3f}\n"
        f"  rank agreement: Spearman ρ = {pooled['spearman_rho']:.2f}, "
        f"Pearson r = {pooled['pearson_r']:.2f}\n"
        f"  selecting on the imagined metric costs {pooled['mean_selection_regret']:.3f} "
        "real success vs selecting on the simulator"
    )


if __name__ == "__main__":
    main()
