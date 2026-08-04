"""Plot real-environment success for evaluated Dream-PPO snapshots.

Consumes the ``results.jsonl`` produced by
``scripts.rq1.evaluate_dream_snapshots`` and creates both PNG and PDF figures.
The default x-axis is imagined env-step equivalents: Dream-PPO training uses no
real environment interaction, while ``env_steps_consumed`` in these checkpoints
can include optional diagnostic simulator evaluations.

Example::

    python -m scripts.rq1.plot_dream_snapshots \
      runs/<dream-run>/snapshot_evaluations/results.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rq_common import (
    INK,
    SERIES_COLORS,
    apply_figure_style,
    compact_steps_formatter,
    read_jsonl,
    save_figure,
    wilson_interval,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_AXES = {
    "imagined_steps": "imagined env-step equivalents",
    "iteration": "PPO iteration",
    "env_steps_consumed": "real environment steps consumed (diagnostic eval)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results",
        type=Path,
        help="snapshot_evaluations/results.jsonl produced by the evaluator",
    )
    parser.add_argument("--x", choices=sorted(X_AXES), default="imagined_steps")
    parser.add_argument(
        "--evaluation-id",
        default=None,
        help="protocol to plot when the JSONL contains more than one protocol",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <results-dir>/dream_snapshot_success.png)",
    )
    parser.add_argument(
        "--title",
        default="Real-environment success across Dream-PPO training",
    )
    return parser


def select_rows(
    rows: list[dict], evaluation_id: str | None = None
) -> tuple[list[dict], str]:
    """Select one protocol and return its rows in training order."""

    usable = [
        row
        for row in rows
        if row.get("evaluation_id") is not None
        and row.get("success_rate") is not None
        and row.get("episodes", 0) > 0
    ]
    if not usable:
        raise ValueError("the JSONL contains no completed snapshot evaluations")

    counts = Counter(str(row["evaluation_id"]) for row in usable)
    if evaluation_id is None:
        # Usually there is one protocol. If a later invocation used different
        # arguments, default to the protocol with the most completed snapshots.
        evaluation_id = max(sorted(counts), key=lambda key: counts[key])
        if len(counts) > 1:
            print(
                f"Multiple evaluation protocols found; plotting {evaluation_id} "
                f"({counts[evaluation_id]} snapshots). Use --evaluation-id to choose another."
            )
    elif evaluation_id not in counts:
        available = ", ".join(f"{key} ({count})" for key, count in sorted(counts.items()))
        raise ValueError(
            f"evaluation protocol {evaluation_id!r} not found; available: {available}"
        )

    selected = [row for row in usable if str(row["evaluation_id"]) == evaluation_id]
    selected.sort(key=lambda row: (row.get("imagined_steps", 0), row["iteration"]))
    return selected, evaluation_id


def confidence_intervals(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for row in rows:
        episodes = int(row["episodes"])
        successes = float(row["success_rate"]) * episodes
        low, high = wilson_interval(successes, episodes)
        lows.append(low)
        highs.append(high)
    return np.asarray(lows), np.asarray(highs)


def plot(rows: list[dict], args: argparse.Namespace, evaluation_id: str) -> Path:
    missing_x = [row["iteration"] for row in rows if row.get(args.x) is None]
    if missing_x:
        raise ValueError(
            f"x-axis field {args.x!r} is missing for iterations {missing_x[:5]}"
        )

    rows = sorted(rows, key=lambda row: (float(row[args.x]), row["iteration"]))
    x = np.asarray([row[args.x] for row in rows], dtype=float)
    success = np.asarray([row["success_rate"] for row in rows], dtype=float)
    ci_low, ci_high = confidence_intervals(rows)

    color = SERIES_COLORS["dream_ppo"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(
        x,
        ci_low,
        ci_high,
        color=color,
        alpha=0.14,
        linewidth=0,
        label="95% Wilson interval",
    )
    ax.plot(x, success, color=color, linewidth=2.2)
    ax.scatter(
        x,
        success,
        s=24,
        color=color,
        edgecolor=INK["surface"],
        linewidth=0.8,
        zorder=3,
        label="canonical simulator evaluation",
    )

    initial = 0
    best = int(np.argmax(success))
    final = len(rows) - 1
    ax.annotate(
        f"BC initialization  {success[initial]:.3f}",
        xy=(x[initial], success[initial]),
        xytext=(12, -24),
        textcoords="offset points",
        color=INK["secondary"],
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK["muted"], "linewidth": 0.8},
    )
    final_label = "best / final" if best == final else "final"
    ax.annotate(
        f"{final_label}  {success[final]:.3f}",
        xy=(x[final], success[final]),
        xytext=(-8, -28),
        textcoords="offset points",
        ha="right",
        color=color,
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9},
    )
    if best != final:
        ax.annotate(
            f"best  {success[best]:.3f}",
            xy=(x[best], success[best]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    protocol = rows[0].get("evaluation", {})
    episodes_per_repeat = protocol.get("episodes", rows[0].get("episodes_per_repeat"))
    repeats = protocol.get("repeats", rows[0].get("repeats"))
    caption = (
        f"protocol {evaluation_id} · {repeats} × {episodes_per_repeat} episodes/snapshot "
        "· fixed-target PushT"
    )
    ax.set_title(args.title, pad=12)
    ax.set_xlabel(X_AXES[args.x])
    ax.set_ylabel("success rate in simulator")
    ax.set_ylim(0.0, 1.0)
    if args.x != "iteration":
        ax.xaxis.set_major_formatter(compact_steps_formatter())
    ax.legend(loc="lower right")
    fig.text(0.5, 0.01, caption, ha="center", color=INK["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    output = (args.output or args.results.parent / "dream_snapshot_success.png").resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("--output must be a .png path; a matching PDF is created automatically")
    save_figure(fig, output)
    plt.close(fig)

    best_row = rows[best]
    final_row = rows[final]
    print(
        f"plotted {len(rows)} snapshots\n"
        f"  initial: iteration {rows[initial]['iteration']}, success {success[initial]:.3f}\n"
        f"  best:    iteration {best_row['iteration']}, success {success[best]:.3f}\n"
        f"  final:   iteration {final_row['iteration']}, success {success[final]:.3f}"
    )
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results_path = args.results.expanduser().resolve()
    if not results_path.is_file():
        raise SystemExit(f"results JSONL not found: {results_path}")
    args.results = results_path

    apply_figure_style()
    try:
        rows, evaluation_id = select_rows(read_jsonl(results_path), args.evaluation_id)
        plot(rows, args, evaluation_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
