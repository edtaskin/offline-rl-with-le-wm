"""Turn the RQ1 ledger into the headline table and the two RQ1 figures.

Reads the ``results.jsonl`` written by :mod:`scripts.rq1.evaluate_grid` and, for
the selection analysis, the ``selection_log.jsonl`` files dropped in dream run
directories. Produces:

* ``table.md``   -- success per agent per training seed, with pooled Wilson CIs
* ``fig_interaction_budget.(png|pdf)`` -- real-PPO success vs env steps consumed,
  with the zero-interaction agents as reference lines at x = 0, and the budget at
  which real PPO first matches dream PPO called out
* ``fig_selection_correlation.(png|pdf)`` -- imagined held-out success vs real
  held-out success at matched checkpoints; the evidence that picking a checkpoint
  without the simulator picks a comparable checkpoint

Usage::

    python -m scripts.rq1.report
    python -m scripts.rq1.report --ledger runs/rq1/results.jsonl --output-dir runs/rq1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


# Validated 3-slot categorical palette (see the dataviz reference palette);
# aqua sits below 3:1 on white, so every series it marks is also direct-labeled.
COLORS = {
    "real-ppo": "#2a78d6",
    "dream-ppo": "#eb6834",
    "bc": "#1baf7a",
}
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SNAPSHOT_PREFIX = "snapshot_step"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default="runs/rq1/results.jsonl")
    parser.add_argument("--output-dir", default="runs/rq1")
    parser.add_argument("--runs-root", default="runs", help="searched for selection_log.jsonl files")
    parser.add_argument(
        "--headline-checkpoint",
        default="best",
        help="checkpoint name that supplies each agent's headline number",
    )
    parser.add_argument("--bc-label", default="bc")
    parser.add_argument("--real-label", default="real-ppo")
    parser.add_argument("--dream-label", default="dream-ppo")
    return parser


# ------------------------------------------------------------------ statistics
def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: the right binomial CI near 0 or 1, where success
    rates in this project live and the normal approximation misbehaves."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation; selection only needs the ordering to survive, not the level."""
    if len(x) < 2:
        return float("nan")
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"no RQ1 ledger at {path}; run scripts/rq1/evaluate_grid.py first"
        )
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def latest_rows(rows: list[dict]) -> list[dict]:
    """Keep the last row per (label, checkpoint, protocol) so re-runs supersede."""
    unique: dict[tuple, dict] = {}
    for row in rows:
        unique[(row["label"], row["checkpoint"], row["protocol"])] = row
    return list(unique.values())


# ----------------------------------------------------------------------- table
def headline_rows(rows: list[dict], label: str, checkpoint_name: str, bc_label: str) -> list[dict]:
    """One row per training seed for ``label``, at its headline checkpoint.

    The budget curve evaluates some checkpoints under a cheaper protocol; if a
    checkpoint was scored under more than one, the richer evaluation wins so a
    headline number is never quietly built from fewer episodes.
    """
    per_seed: dict[object, dict] = {}
    for row in rows:
        if row["label"] != label:
            continue
        # BC has a single checkpoint; the PPO agents report the selected one.
        if label != bc_label and row["checkpoint_name"] != checkpoint_name:
            continue
        seed = row["seed"]
        if seed not in per_seed or row["episodes"] > per_seed[seed]["episodes"]:
            per_seed[seed] = row
    return sorted(per_seed.values(), key=lambda r: (r["seed"] is None, r["seed"]))


def headline_table(rows: list[dict], checkpoint_name: str, bc_label: str) -> str:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for label in {row["label"] for row in rows}:
        entries = headline_rows(rows, label, checkpoint_name, bc_label)
        if entries:
            by_label[label] = entries

    lines = [
        f"## Headline: success rate at checkpoint `{checkpoint_name}`",
        "",
        "| agent | seeds | per-seed success | pooled | 95% CI (Wilson) | episodes | env steps consumed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in sorted(by_label):
        entries = by_label[label]
        rates = [entry["success_rate"] for entry in entries]
        episodes = sum(entry["episodes"] for entry in entries)
        successes = sum(entry["success_rate"] * entry["episodes"] for entry in entries)
        low, high = wilson_interval(successes, episodes)
        budgets = {entry["env_steps_consumed"] for entry in entries}
        budget = (
            "unknown"
            if None in budgets
            else " / ".join(f"{value:,}" for value in sorted(budgets))
        )
        per_seed = ", ".join(
            f"s{entry['seed']}={entry['success_rate']:.3f}" for entry in entries
        )
        spread = f" ± {np.std(rates):.3f}" if len(rates) > 1 else ""
        lines.append(
            f"| {label} | {len(entries)} | {per_seed} | "
            f"{np.mean(rates):.3f}{spread} | [{low:.3f}, {high:.3f}] | {episodes} | {budget} |"
        )
    lines += [
        "",
        "Pooled = all episodes from all seeds; the ± is the spread across training "
        "seeds, which is the quantity that matters for a claim about the method "
        "rather than about one run.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- figures
def _style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def budget_curve(rows: list[dict], real_label: str) -> dict[int, list[tuple[int, float]]]:
    """Per-seed ``[(env_steps, success), ...]`` from the step-tagged snapshots."""
    curves: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["label"] != real_label or not row["checkpoint_name"].startswith(SNAPSHOT_PREFIX):
            continue
        if row["env_steps_consumed"] is None:
            continue
        curves[row["seed"]].append((row["env_steps_consumed"], row["success_rate"]))
    return {seed: sorted(points) for seed, points in curves.items()}


def crossing_budget(steps: np.ndarray, success: np.ndarray, level: float) -> float | None:
    """First env-step budget at which the real-PPO mean curve reaches ``level``."""
    for index in range(len(steps)):
        if success[index] >= level:
            if index == 0:
                return float(steps[0])
            # Linear interpolation between the bracketing evaluated budgets.
            x0, x1 = steps[index - 1], steps[index]
            y0, y1 = success[index - 1], success[index]
            if y1 == y0:
                return float(x1)
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return None


def sustained_crossing_budget(steps: np.ndarray, success: np.ndarray, level: float) -> float | None:
    """First budget from which the curve stays at or above ``level`` for good.

    PPO's learning curve is not monotone, so the first crossing can be a spike
    the run then falls back below. Quoting only that number overstates how little
    interaction the baseline needs; this is the honest companion to it.
    """
    above = success >= level
    if not above.any() or not above[-1]:
        return None
    index = len(above) - 1
    while index > 0 and above[index - 1]:
        index -= 1
    return float(steps[index])


def plot_interaction_budget(rows, output_dir: Path, args) -> None:
    curves = budget_curve(rows, args.real_label)
    if not curves:
        print("! no real-PPO snapshots in the ledger; skipping the budget figure")
        return

    def level(label: str) -> tuple[float, float, float] | None:
        entries = headline_rows(rows, label, args.headline_checkpoint, args.bc_label)
        if not entries:
            return None
        rates = [entry["success_rate"] for entry in entries]
        return float(np.mean(rates)), float(np.min(rates)), float(np.max(rates))

    dream = level(args.dream_label)
    bc = level(args.bc_label)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    _style_axes(ax)

    # Real-PPO: per-seed curves on a shared budget grid, summarized by the mean
    # with a min-max band (with three seeds, the observed range is more honest
    # than a standard deviation).
    grid = np.array(sorted({step for points in curves.values() for step, _ in points}))
    stacked = []
    for points in curves.values():
        steps = np.array([step for step, _ in points], dtype=float)
        values = np.array([value for _, value in points], dtype=float)
        stacked.append(np.interp(grid, steps, values))
    stacked = np.vstack(stacked)
    mean = stacked.mean(axis=0)

    if len(curves) > 1:
        ax.fill_between(
            grid, stacked.min(axis=0), stacked.max(axis=0),
            color=COLORS["real-ppo"], alpha=0.15, linewidth=0, zorder=2,
        )
    ax.plot(
        grid, mean, color=COLORS["real-ppo"], linewidth=2.0, zorder=4,
        marker="o", markersize=5, markerfacecolor=COLORS["real-ppo"],
        markeredgecolor="white", markeredgewidth=1.0,
        label=f"PPO in the simulator ({len(curves)} seeds)",
    )

    for label, stats, color in (
        ("PPO in imagination — 0 env steps", dream, COLORS["dream-ppo"]),
        ("BC (expert data only)", bc, COLORS["bc"]),
    ):
        if stats is None:
            continue
        value, low, high = stats
        ax.axhline(value, color=color, linewidth=2.0, linestyle="--", zorder=3, label=label)
        if high > low:
            ax.fill_between(
                [0, grid.max()], [low, low], [high, high],
                color=color, alpha=0.12, linewidth=0, zorder=1,
            )
        ax.plot([0], [value], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        # Direct label: required relief for the sub-3:1 aqua, and it keeps the
        # two reference levels legible without a legend round-trip. Anchored at
        # the left, where the rising curve and the crossing callout are not.
        ax.annotate(
            f"{label}  {value:.2f}",
            xy=(grid.min(), value), xytext=(6, 5), textcoords="offset points",
            ha="left", va="bottom", color=INK, fontsize=9, fontweight="medium",
        )

    if dream is not None:
        crossing = crossing_budget(grid, mean, dream[0])
        sustained = sustained_crossing_budget(grid, mean, dream[0])
        if crossing is not None:
            ax.axvline(crossing, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
            caption = f"real PPO first matches\nimagination: {crossing:,.0f} env steps"
            if sustained is not None and sustained > crossing:
                # The learning curve dips back below in between, so both numbers
                # are needed to describe the baseline's real cost.
                ax.axvline(sustained, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
                caption += f"\nstays above it: {sustained:,.0f}"
            ax.annotate(
                caption,
                xy=(crossing, dream[0]), xytext=(8, -34), textcoords="offset points",
                ha="left", va="top", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
            )

    ax.set_xlabel("real environment steps consumed (training + checkpoint selection)", fontsize=10, color=INK)
    ax.set_ylabel("success rate", fontsize=10, color=INK)
    ax.set_title(
        "Policy improvement without touching the environment",
        fontsize=12, color=INK, fontweight="semibold", loc="left", pad=12,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_interaction_budget.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_dir}/fig_interaction_budget.png")


def load_selection_logs(runs_root: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_root.glob("**/selection_log.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    row["run"] = str(path.parent)
                    rows.append(row)
    return rows


def selection_pairs(rows: list[dict], runs_root: Path, dream_label: str) -> list[dict]:
    """Pair imagined held-out success with real success at the same checkpoint.

    Two sources, in order of quality. Primarily a *post hoc* join: dream
    snapshots evaluated by the canonical evaluator, matched by iteration to the
    imagined score recorded during training. That keeps the training run itself
    at zero env steps -- the real numbers are produced by analysis afterwards,
    never by selection. Falls back to rows where a training run happened to log
    both (``--eval_interval`` on a dream run), which cost interaction.
    """
    logged = {}
    for row in load_selection_logs(runs_root):
        if row.get("dream_success") is not None:
            logged[(row["run"], row["iteration"])] = row

    pairs = []
    for row in rows:
        if row["label"] != dream_label or not row["checkpoint_name"].startswith(SNAPSHOT_PREFIX):
            continue
        match = re.search(r"_it(\d+)$", row["checkpoint_name"])
        if not match:
            continue
        key = (str(Path(row["checkpoint"]).parent), int(match.group(1)))
        entry = logged.get(key)
        if entry is None:
            continue
        pairs.append(
            {
                "dream_success": entry["dream_success"],
                "real_success": row["success_rate"],
                "source": "post-hoc",
            }
        )

    if not pairs:
        pairs = [
            {
                "dream_success": row["dream_success"],
                "real_success": row["real_success"],
                "source": "in-training",
            }
            for row in logged.values()
            if row.get("real_success") is not None
        ]
    return pairs


def plot_selection_correlation(rows: list[dict], runs_root: Path, output_dir: Path, dream_label: str) -> None:
    pairs = selection_pairs(rows, runs_root, dream_label)
    if len(pairs) < 3:
        print("! fewer than 3 paired dream/real evaluations; skipping the selection figure")
        print("  (evaluate the dream runs' snapshot_step*.pt checkpoints to produce them)")
        return

    dream = np.array([pair["dream_success"] for pair in pairs])
    real = np.array([pair["real_success"] for pair in pairs])
    rho = spearman(dream, real)
    source = pairs[0]["source"]

    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=200)
    _style_axes(ax)
    ax.scatter(
        dream, real, s=48, color=COLORS["dream-ppo"], alpha=0.85,
        edgecolors="white", linewidths=1.0, zorder=4,
    )
    ax.set_xlabel("imagined held-out success (0 env steps)", fontsize=10, color=INK)
    ax.set_ylabel("real held-out success", fontsize=10, color=INK)
    ax.set_title(
        f"Does interaction-free selection track the real thing?\n"
        f"Spearman ρ = {rho:.2f} over {len(pairs)} checkpoints ({source})",
        fontsize=11, color=INK, fontweight="semibold", loc="left", pad=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_selection_correlation.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_dir}/fig_selection_correlation.png (Spearman rho = {rho:.3f})")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = latest_rows(load_ledger(Path(args.ledger)))
    table = headline_table(rows, args.headline_checkpoint, args.bc_label)
    (output_dir / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote {output_dir}/table.md")

    plot_interaction_budget(rows, output_dir, args)
    plot_selection_correlation(rows, Path(args.runs_root), output_dir, args.dream_label)


if __name__ == "__main__":
    main()
