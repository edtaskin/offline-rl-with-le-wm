"""Poster-ready figures for RQ1, generated from the evaluation ledger.

Four figures, in the order a reader who knows nothing about the project needs
them:

1. ``fig1_concept``    -- what "training inside a world model" even means
2. ``fig2_headline``   -- the three success rates, with what each one cost
3. ``fig3_budget``     -- how much simulation the imagined training replaces
4. ``fig4_selection``  -- the part that did not work, stated plainly

Numbers come from ``runs/rq1/results.jsonl`` and the dream runs' selection logs;
nothing here is hard-coded, so re-running after more seeds updates every figure
and the accompanying ``poster_tables.md``.

Usage::

    python -m scripts.rq1.poster_figures
    python -m scripts.rq1.poster_figures --headline-checkpoint best
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from scripts.rq1.report import (
    COLORS,
    GRID,
    INK,
    INK_MUTED,
    AXIS,
    SNAPSHOT_PREFIX,
    budget_curve,
    crossing_budget,
    headline_rows,
    latest_rows,
    load_ledger,
    load_selection_logs,
    selection_pairs,
    spearman,
    sustained_crossing_budget,
    wilson_interval,
)

# Poster figures are read from ~1.5 m away; everything is a step larger than the
# report defaults, and every figure ships as PDF too so it scales without loss.
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
ANNOTATION_SIZE = 11
DPI = 300

AGENTS = (
    ("bc", "Behavioural cloning", "copy the expert"),
    ("dream-ppo", "PPO in imagination", "practise in the world model"),
    ("real-ppo", "PPO in the simulator", "practise in the real simulator"),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default="runs/rq1/results.jsonl")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="runs/rq1/poster")
    parser.add_argument(
        "--headline-checkpoint",
        default="final",
        help="'final' is the RQ1 headline: no checkpoint selection, so the dream "
        "agent's interaction budget stays exactly zero",
    )
    return parser


def _save(fig, output_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight", facecolor="white", dpi=DPI)
    plt.close(fig)
    print(f"  wrote {output_dir}/{name}.png (+ .pdf)")


def _titles(fig, ax, main: str, subtitle: str) -> None:
    """Bold headline above a muted methodology line.

    Both go through the title machinery rather than free ``ax.text``: a long text
    artist that overhangs the axes makes ``tight_layout`` shrink the plot to fit
    it, which is what squeezes a chart into half its canvas.
    """
    fig.suptitle(main, fontsize=TITLE_SIZE + 2, color=INK, fontweight="bold", x=0.02, ha="left")
    ax.set_title(subtitle, fontsize=ANNOTATION_SIZE - 1, color=INK_MUTED, loc="left", pad=10)


def _style_axes(ax, *, grid_axis="both"):
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(colors=INK_MUTED, labelsize=TICK_SIZE)


# ------------------------------------------------------ 1. the concept figure
def figure_concept(output_dir: Path) -> None:
    """Two ways to improve a policy, drawn side by side.

    A poster reader's first question is not "how well does it work" but "what is
    the thing you did"; the whole result is only surprising once you can see that
    the right-hand loop never touches the simulator.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))

    def box(ax, xy, width, height, text, color, *, bold=True):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                xy, width, height,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=2.0, edgecolor=color, facecolor=color, alpha=0.12, zorder=2,
            )
        )
        ax.text(
            xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=LABEL_SIZE, color=INK, zorder=3,
            fontweight="semibold" if bold else "normal", linespacing=1.4,
        )

    def arrow(ax, start, end, color, *, rad=0.0):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=2.2, mutation_scale=18,
                connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3,
            ),
            zorder=4,
        )

    def panel(ax, title, color, cost, bottom_text, footnote=None):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=TITLE_SIZE, color=INK, fontweight="semibold", pad=14)

        box(ax, (1.4, 8.4), 7.2, 1.0, "Recorded expert demonstrations", COLORS["bc"])
        arrow(ax, (5.0, 8.4), (5.0, 7.6), COLORS["bc"])
        ax.text(5.25, 8.0, "starts the policy off", ha="left", va="center",
                fontsize=ANNOTATION_SIZE, color=INK_MUTED)

        box(ax, (2.3, 5.8), 5.4, 1.8, "Policy\n(what controls the robot)", color)
        box(ax, (2.3, 1.9), 5.4, 1.8, bottom_text, color)

        # The practice loop: act, then receive the consequence.
        arrow(ax, (3.4, 5.8), (3.4, 3.7), color, rad=0.32)
        ax.text(2.05, 4.75, "action", ha="right", va="center",
                fontsize=ANNOTATION_SIZE, color=INK_MUTED)
        arrow(ax, (6.6, 3.7), (6.6, 5.8), color, rad=0.32)
        ax.text(7.95, 4.75, "next picture\n+ reward", ha="left", va="center",
                fontsize=ANNOTATION_SIZE, color=INK_MUTED, linespacing=1.4)

        if footnote:
            ax.text(5.0, 1.35, footnote, ha="center", va="center",
                    fontsize=ANNOTATION_SIZE - 1, color=INK_MUTED, style="italic")
        ax.text(5.0, 0.45, cost, ha="center", va="center",
                fontsize=LABEL_SIZE + 1, color=color, fontweight="bold")

    panel(
        axes[0], "The usual way: practise in the simulator", COLORS["real-ppo"],
        "costs ~1,000,000 simulator steps", "PushT simulator",
    )
    panel(
        axes[1], "This work: practise inside a learned world model", COLORS["dream-ppo"],
        "costs 0 simulator steps", "World model (frozen)\n+ success detector",
        footnote="built once from the same expert videos",
    )

    fig.suptitle(
        "Can a policy get better by practising only in its own imagination?",
        fontsize=TITLE_SIZE + 3, color=INK, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_dir, "fig1_concept")


# ---------------------------------------------------- 2. the headline figure
def headline_stats(rows, checkpoint_name: str) -> dict:
    stats = {}
    for label, _title, _gloss in AGENTS:
        entries = headline_rows(rows, label, checkpoint_name, "bc")
        if not entries:
            continue
        episodes = sum(entry["episodes"] for entry in entries)
        successes = sum(entry["success_rate"] * entry["episodes"] for entry in entries)
        budgets = [entry["env_steps_consumed"] for entry in entries]
        stats[label] = {
            "rates": [entry["success_rate"] for entry in entries],
            "pooled": successes / episodes,
            "ci": wilson_interval(successes, episodes),
            "episodes": episodes,
            "seeds": len(entries),
            "budget": None if None in budgets else float(np.mean(budgets)),
        }
    return stats


def figure_headline(stats: dict, output_dir: Path) -> None:
    """The three numbers, with the cost of each printed under its bar."""
    present = [(label, title, gloss) for label, title, gloss in AGENTS if label in stats]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    _style_axes(ax, grid_axis="y")

    positions = np.arange(len(present))
    values = [stats[label]["pooled"] for label, _, _ in present]
    colors = [COLORS[label] for label, _, _ in present]
    errors = np.array(
        [
            [stats[label]["pooled"] - stats[label]["ci"][0] for label, _, _ in present],
            [stats[label]["ci"][1] - stats[label]["pooled"] for label, _, _ in present],
        ]
    )

    ax.bar(positions, values, width=0.56, color=colors, zorder=3)
    ax.errorbar(
        positions, values, yerr=errors, fmt="none",
        ecolor=INK, elinewidth=1.6, capsize=7, capthick=1.6, zorder=4,
    )

    # Per-seed points beside each bar: three runs, shown rather than summarized.
    for position, (label, _, _) in zip(positions, present):
        rates = stats[label]["rates"]
        if len(rates) > 1:
            ax.scatter(
                np.full(len(rates), position + 0.34), rates,
                s=34, color=INK, alpha=0.55, zorder=5, linewidths=0,
            )

    for position, value, (label, _, _) in zip(positions, values, present):
        ax.text(
            position, value + errors[1][list(positions).index(position)] + 0.035,
            f"{value:.0%}", ha="center", va="bottom",
            fontsize=TITLE_SIZE + 2, color=INK, fontweight="bold",
        )

    # Two short lines only: the agent, and what it cost. The prose explanation
    # lives beside the figure, not inside it.
    labels = []
    for label, title, _gloss in present:
        budget = stats[label]["budget"]
        cost = "0 simulator steps" if not budget else f"{budget/1e6:.2f}M simulator steps"
        # Every label is two lines so the cost row sits at a common baseline.
        wrapped = {
            "bc": "Behavioural\ncloning",
            "dream-ppo": "PPO\nin imagination",
            "real-ppo": "PPO\nin the simulator",
        }.get(label, title)
        labels.append(f"{wrapped}\n\n{cost}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=LABEL_SIZE, color=INK, linespacing=1.5)
    ax.tick_params(axis="x", length=0)

    ax.set_ylabel("tasks solved (success rate)", fontsize=LABEL_SIZE, color=INK)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    episodes = max(stats[label]["episodes"] for label, _, _ in present)
    seeds = max(stats[label]["seeds"] for label, _, _ in present)
    _titles(
        fig, ax,
        "Practising in imagination recovers most of the gain — for free",
        f"PushT block-pushing · {episodes} test episodes per agent · up to {seeds} training runs each\n"
        "bars are 95% confidence intervals · grey dots are the individual runs",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_dir, "fig2_headline")


# ------------------------------------------------------- 3. the budget figure
def figure_budget(rows, stats: dict, output_dir: Path) -> None:
    curves = budget_curve(rows, "real-ppo")
    if not curves:
        print("  ! no real-PPO snapshots; skipping fig3")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    _style_axes(ax)

    grid = np.array(sorted({step for points in curves.values() for step, _ in points}))
    stacked = np.vstack(
        [
            np.interp(grid, [s for s, _ in points], [v for _, v in points])
            for points in curves.values()
        ]
    )
    mean = stacked.mean(axis=0)

    ax.fill_between(
        grid, stacked.min(axis=0), stacked.max(axis=0),
        color=COLORS["real-ppo"], alpha=0.15, linewidth=0, zorder=2,
    )
    ax.plot(
        grid, mean, color=COLORS["real-ppo"], linewidth=2.6, zorder=4,
        marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2,
        label=f"PPO practising in the simulator ({len(curves)} runs)",
    )

    for label, key, text in (
        ("dream-ppo", "dream-ppo", "PPO practising in imagination — 0 simulator steps"),
        ("bc", "bc", "Behavioural cloning — copies the expert, 0 simulator steps"),
    ):
        if key not in stats:
            continue
        value = stats[key]["pooled"]
        ax.axhline(value, color=COLORS[label], linewidth=2.6, linestyle="--", zorder=3, label=text)
        ax.plot([0], [value], marker="o", markersize=10, color=COLORS[label],
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.annotate(
            f"{value:.0%}", xy=(grid.min(), value), xytext=(8, 7), textcoords="offset points",
            ha="left", va="bottom", color=INK, fontsize=LABEL_SIZE, fontweight="bold",
        )

    if "dream-ppo" in stats:
        level = stats["dream-ppo"]["pooled"]
        sustained = sustained_crossing_budget(grid, mean, level)
        first = crossing_budget(grid, mean, level)
        marker = sustained if sustained is not None else first
        if marker is not None:
            ax.axvline(marker, color=INK_MUTED, linewidth=1.4, linestyle=":", zorder=2)
            ax.annotate(
                f"the simulator needs ~{marker/1000:,.0f}k steps\nto reach what imagination got for free",
                xy=(marker, level), xytext=(18, -46), textcoords="offset points",
                ha="left", va="top", color=INK, fontsize=ANNOTATION_SIZE,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.0),
            )

    ax.set_xlabel("simulator steps used (training + checkpoint selection)", fontsize=LABEL_SIZE, color=INK)
    ax.set_ylabel("tasks solved (success rate)", fontsize=LABEL_SIZE, color=INK)
    _titles(
        fig, ax,
        "How much simulation does imagined practice replace?",
        "each point is a saved checkpoint, re-tested from scratch · shaded band spans the 3 training runs",
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    legend = ax.legend(loc="lower right", frameon=False, fontsize=ANNOTATION_SIZE)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_dir, "fig3_budget")


# ---------------------------------------------------- 4. the honest caveat
def figure_selection(rows, runs_root: Path, output_dir: Path) -> None:
    pairs = selection_pairs(rows, runs_root, "dream-ppo")
    if len(pairs) < 3:
        print("  ! not enough paired evaluations; skipping fig4")
        return

    dream = np.array([pair["dream_success"] for pair in pairs])
    real = np.array([pair["real_success"] for pair in pairs])
    rho = spearman(dream, real)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    _style_axes(ax)
    # Equal aspect over equal ranges, so the parity line really is at 45 degrees
    # and the label sitting on it can be rotated to match.
    ax.set_aspect("equal", adjustable="box")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.4, linestyle="--", zorder=2)
    # Runs back down the parity line from this anchor, through the empty
    # upper-left triangle rather than across the points or the tick labels.
    ax.annotate(
        "if imagination were honest, points would lie here",
        xy=(0.84, 0.84), xytext=(-6, 7), textcoords="offset points",
        ha="right", va="bottom", color=INK_MUTED, fontsize=ANNOTATION_SIZE - 1,
        rotation=45, rotation_mode="anchor",
    )
    ax.scatter(
        dream, real, s=80, color=COLORS["dream-ppo"], alpha=0.85,
        edgecolors="white", linewidths=1.3, zorder=4,
    )

    ax.set_xlabel("how good the policy THINKS it is\n(success in imagination)", fontsize=LABEL_SIZE, color=INK)
    ax.set_ylabel("how good it actually is\n(success in the simulator)", fontsize=LABEL_SIZE, color=INK)
    _titles(
        fig, ax,
        "What did not work: imagination is a poor judge of itself",
        f"{len(pairs)} checkpoints · rank correlation ρ = {rho:.2f}\n"
        "every point sits below the line — imagination is consistently over-optimistic",
    )
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_dir, "fig4_selection")
    return rho


# ------------------------------------------------------------------- tables
def write_tables(rows, stats: dict, runs_root: Path, output_dir: Path, checkpoint_name: str) -> None:
    lines = [
        "# RQ1 poster tables",
        "",
        "## Table 1 — Main result",
        "",
        "| Agent | What it learns from | Simulator steps used | Tasks solved | 95% CI | Test episodes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    sources = {
        "bc": "expert demonstrations only",
        "dream-ppo": "expert demonstrations + practice inside the world model",
        "real-ppo": "expert demonstrations + practice in the simulator",
    }
    for label, title, _gloss in AGENTS:
        if label not in stats:
            continue
        entry = stats[label]
        budget = "**0**" if not entry["budget"] else f"{entry['budget']:,.0f}"
        low, high = entry["ci"]
        lines.append(
            f"| {title} | {sources[label]} | {budget} | **{entry['pooled']:.1%}** | "
            f"{low:.1%} – {high:.1%} | {entry['episodes']} |"
        )

    lines += [
        "",
        f"Checkpoint: `{checkpoint_name}.pt` (the policy at the end of training, so no "
        "checkpoint is chosen using the simulator). Evaluation is identical for all three "
        "agents: 50 episodes x 3 repeats per training run, on held-out episode seeds.",
        "",
        "## Table 2 — Where the simulator steps actually go",
        "",
        "| Agent | Steps for training | Steps for choosing a checkpoint | Total |",
        "| --- | --- | --- | --- |",
    ]
    for label, title, _gloss in AGENTS:
        entries = headline_rows(rows, label, checkpoint_name, "bc")
        if not entries:
            continue
        train = [e.get("train_env_steps") for e in entries]
        select = [e.get("eval_env_steps") for e in entries]
        total = [e.get("env_steps_consumed") for e in entries]
        if None in train or None in select:
            continue
        lines.append(
            f"| {title} | {np.mean(train):,.0f} | {np.mean(select):,.0f} | {np.mean(total):,.0f} |"
        )
    lines += [
        "",
        "Choosing which saved checkpoint to keep is itself paid for in simulator steps. "
        "The imagined agent pays neither column.",
        "",
        "## Table 3 — Per-run results",
        "",
        "| Agent | Run 1 | Run 2 | Run 3 | Spread |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, title, _gloss in AGENTS:
        if label not in stats:
            continue
        rates = stats[label]["rates"]
        cells = [f"{rate:.1%}" for rate in rates] + ["–"] * (3 - len(rates))
        spread = f"± {np.std(rates):.1%}" if len(rates) > 1 else "single run"
        lines.append(f"| {title} | {cells[0]} | {cells[1]} | {cells[2]} | {spread} |")
    lines += [
        "",
        "Each run is an independent training seed. Behavioural cloning is a single "
        "published checkpoint shared by all runs, so it has no spread.",
        "",
    ]

    pairs = selection_pairs(rows, runs_root, "dream-ppo")
    if len(pairs) >= 3:
        dream = np.array([pair["dream_success"] for pair in pairs])
        real = np.array([pair["real_success"] for pair in pairs])
        lines += [
            "## Table 4 — The negative result",
            "",
            "| Quantity | Value |",
            "| --- | --- |",
            f"| Checkpoints compared | {len(pairs)} |",
            f"| Rank correlation (imagined vs real success) | ρ = {spearman(dream, real):.2f} |",
            f"| Success in imagination | {dream.min():.0%} – {dream.max():.0%} |",
            f"| Success in the simulator | {real.min():.0%} – {real.max():.0%} |",
            "",
            "Imagined success is both too high and too flat to pick a checkpoint with. "
            "Taking the last checkpoint instead is simpler and scored better.",
            "",
        ]

    path = output_dir / "poster_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = latest_rows(load_ledger(Path(args.ledger)))
    stats = headline_stats(rows, args.headline_checkpoint)
    print(f"RQ1 poster figures -> {output_dir}")

    figure_concept(output_dir)
    figure_headline(stats, output_dir)
    figure_budget(rows, stats, output_dir)
    figure_selection(rows, Path(args.runs_root), output_dir)
    write_tables(rows, stats, Path(args.runs_root), output_dir, args.headline_checkpoint)


if __name__ == "__main__":
    main()
