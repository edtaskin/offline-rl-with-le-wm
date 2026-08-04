"""RQ1 poster assets: the headline table and the two figures.

Reads what ``evaluate_grid.py`` and ``budget_curve.py`` measured and turns it
into ``table.md``, ``summary.json`` and two figures:

* ``rq1_success.png``  -- pooled success per agent, with the per-seed points
  drawn on top so the seed spread is visible rather than hidden in an error bar.
* ``rq1_budget_curve.png`` -- real-env PPO success against environment steps
  consumed, with dream PPO as a horizontal line at x = 0. The line is dream
  PPO's ``final.pt``: no checkpoint selection, hence no interaction at all.

The crossover -- the interaction budget at which real-env PPO first reaches, and
then stays above, the dream level -- is computed here and printed, because "how
many env steps is the world model worth" is the sentence the figure has to
support.

Usage::

    python -m scripts.rq1.report
    python -m scripts.rq1.report --dream-variant best   # dream-selected instead
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    compact_steps_formatter,
    read_jsonl,
    repo_path,
    save_figure,
    wilson_interval,
)

VARIANT_LABELS = {
    "published": "published checkpoint",
    "final": "final.pt (no selection)",
    "best": "best.pt (own selection rule)",
    "best_real_sel": "best by real-env eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-root", default="runs/rq1")
    parser.add_argument("--output-root", default="runs/rq1/figures")
    parser.add_argument(
        "--dream-variant",
        default="final",
        choices=["final", "best"],
        help="which dream checkpoint is the headline / the x=0 line (default: final)",
    )
    parser.add_argument(
        "--real-variant",
        default="final",
        choices=["final", "best"],
        help="which real-env PPO checkpoint is the headline (default: final)",
    )
    return parser.parse_args()


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["variant"])].append(row)
    return grouped


def pooled(rows: list[dict]) -> dict:
    """Pool equally sized per-seed evaluations into one estimate.

    Every row has the same episode count, so the pooled rate is the mean of the
    per-seed rates; the CI is computed on the pooled episode count, and the
    across-seed std is reported alongside because it answers a different question
    (run-to-run variability, not sampling noise).
    """
    successes = np.array([row["success_rate"] for row in rows], dtype=float)
    episodes = int(sum(row["episodes"] for row in rows))
    rate = float(successes.mean())
    low, high = wilson_interval(rate * episodes, episodes)
    # required_env_steps is what the checkpoint had to buy; env_steps_consumed
    # can be larger for a dream run whose diagnostic real eval was switched on.
    required = [row.get("required_env_steps", row.get("env_steps_consumed", 0)) or 0 for row in rows]
    diagnostic = [row.get("diagnostic_env_steps", 0) or 0 for row in rows]
    return {
        "success_rate": rate,
        "std_across_seeds": float(successes.std(ddof=1)) if len(successes) > 1 else 0.0,
        "ci95": [low, high],
        "episodes": episodes,
        "seeds": [row["seed"] for row in rows],
        "per_seed_success": successes.tolist(),
        "median_required_env_steps": int(np.median(required)) if required else 0,
        "required_env_steps_range": [int(min(required)), int(max(required))] if required else [0, 0],
        "median_diagnostic_env_steps": int(np.median(diagnostic)) if diagnostic else 0,
        "train_env_steps": [row.get("train_env_steps", 0) for row in rows],
        "eval_env_steps": [row.get("eval_env_steps", 0) for row in rows],
        "interaction_free": all(row.get("interaction_free", False) for row in rows),
    }


def write_table(grouped, path: Path, args) -> dict:
    summary = {}
    lines = [
        "# RQ1 — policy improvement inside the world model",
        "",
        "Canonical evaluator (`src.evaluation.evaluate_pusht`), "
        f"{args.repeats_note}, block starts within 200 px of the goal.",
        "",
        "| Agent | Checkpoint | Interaction-free | Success | 95% CI | Across-seed std | Episodes | Env steps required (median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = [
        ("bc", "published"),
        ("real_ppo", "final"),
        ("real_ppo", "best"),
        ("dream_ppo", "final"),
        ("dream_ppo", "best"),
        ("dream_ppo", "best_real_sel"),
    ]
    for key in order:
        if key not in grouped:
            continue
        method, variant = key
        stats = pooled(grouped[key])
        summary[f"{method}/{variant}"] = stats
        lines.append(
            f"| {SERIES_LABELS.get(method, method)} | {VARIANT_LABELS.get(variant, variant)} "
            f"| {'yes' if stats['interaction_free'] else 'no'} "
            f"| **{stats['success_rate']:.3f}** "
            f"| [{stats['ci95'][0]:.3f}, {stats['ci95'][1]:.3f}] "
            f"| {stats['std_across_seeds']:.3f} "
            f"| {stats['episodes']} "
            f"| {stats['median_required_env_steps']:,} |"
        )

    seeds = sorted(
        {row["seed"] for rows in grouped.values() for row in rows if row["seed"] is not None}
    )
    lines += [
        "",
        "Per-seed success rates:",
        "",
        "| Agent | Checkpoint | " + " | ".join(f"seed {s}" for s in seeds) + " |",
        "|---|---|" + "---|" * len(seeds),
    ]
    for key in order:
        if key not in grouped:
            continue
        method, variant = key
        by_seed = {row["seed"]: row["success_rate"] for row in grouped[key]}
        if list(by_seed) == [None]:
            # BC is seed-independent: one published checkpoint backs every run.
            cells = [f"{by_seed[None]:.3f} (shared)"] + ["←"] * (len(seeds) - 1)
        else:
            cells = [f"{by_seed[s]:.3f}" if s in by_seed else "n/a" for s in seeds]
        lines.append(
            f"| {SERIES_LABELS.get(method, method)} | {VARIANT_LABELS.get(variant, variant)} | "
            + " | ".join(cells)
            + " |"
        )

    lines += [
        "",
        "**Caveat.** All seeds share the one published BC checkpoint, so the",
        "across-seed spread covers PPO / dream-rollout RNG only, not variation in",
        "the BC prior.",
        "",
        "**On the interaction budget.** `env steps required` = rollout steps +",
        "steps spent on real-env checkpoint selection, because selecting on the",
        "simulator is environment interaction even though it trains nothing. Dream",
        "PPO with `--selection dream` requires zero on both counts; the real-env",
        "steps its run also spent come from `--record-real-eval`, which is RQ2",
        "instrumentation, never feeds selection, and is excluded here (reported as",
        "`median_diagnostic_env_steps` in `summary.json`).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return summary


def plot_success(grouped, args, output_root: Path) -> None:
    bars = [
        ("bc", "published"),
        ("real_ppo", args.real_variant),
        ("dream_ppo", args.dream_variant),
    ]
    bars = [key for key in bars if key in grouped]
    if not bars:
        print("no rows to plot for rq1_success")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    tick_labels = []
    multi_seed = False
    for index, key in enumerate(bars):
        method, _ = key
        stats = pooled(grouped[key])
        multi_seed = multi_seed or len(stats["per_seed_success"]) > 1
        # The budget is read off the data, never assumed: a run stopped early or
        # extended would otherwise be captioned with a number it never paid.
        required = stats["median_required_env_steps"]
        if required == 0:
            budget_label = "0 env steps"
        elif required >= 1e6:
            budget_label = f"{required / 1e6:.2f}M env steps"
        else:
            budget_label = f"{required / 1e3:.0f}k env steps"
        tick_labels.append(f"{SERIES_LABELS[method]}\n{budget_label}")
        color = SERIES_COLORS[method]
        ax.bar(
            index,
            stats["success_rate"],
            width=0.56,
            color=color,
            edgecolor=INK["surface"],
            linewidth=2,  # 2px surface gap between adjacent fills
        )
        ax.errorbar(
            index,
            stats["success_rate"],
            yerr=[
                [stats["success_rate"] - stats["ci95"][0]],
                [stats["ci95"][1] - stats["success_rate"]],
            ],
            fmt="none",
            ecolor=INK["secondary"],
            elinewidth=1.2,
            capsize=4,
        )
        # Direct value labels: identity is never colour-alone, and they discharge
        # the low-contrast relief rule for the aqua slot.
        ax.text(
            index,
            stats["ci95"][1] + 0.035,
            f"{stats['success_rate']:.2f}",
            ha="center",
            va="bottom",
            color=INK["primary"],
            fontweight="bold",
        )
        per_seed = stats["per_seed_success"]
        if len(per_seed) > 1:
            jitter = np.linspace(-0.13, 0.13, len(per_seed))
            ax.plot(
                index + jitter,
                per_seed,
                "o",
                markersize=5,
                color=INK["surface"],
                markeredgecolor=INK["secondary"],
                markeredgewidth=1.4,
                zorder=3,
            )

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("PushT success rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Policy improvement without environment interaction", pad=12)
    ax.grid(axis="x", visible=False)
    caption = "bars = pooled over seeds, 95% Wilson CI"
    if multi_seed:
        caption = "hollow dots = individual training seeds · " + caption
    fig.text(0.5, -0.06, caption, ha="center", color=INK["muted"], fontsize=8)
    save_figure(fig, output_root / "rq1_success.png")
    plt.close(fig)


def plot_budget_curve(grouped, curve_rows, args, output_root: Path) -> dict:
    real_rows = [row for row in curve_rows if row["method"] == "real_ppo"]
    if not real_rows:
        print("no real-PPO snapshot evaluations; run scripts/rq1/budget_curve.py first")
        return {}

    dream_key = ("dream_ppo", args.dream_variant)
    dream_level = pooled(grouped[dream_key])["success_rate"] if dream_key in grouped else None

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in real_rows:
        by_seed[row["seed"]].append(row)

    all_steps = sorted({row["env_steps_consumed"] for row in real_rows})
    per_seed_curves = []
    for _seed, rows in sorted(by_seed.items()):
        rows = sorted(rows, key=lambda row: row["env_steps_consumed"])
        steps = np.array([row["env_steps_consumed"] for row in rows], dtype=float)
        success = np.array([row["success_rate"] for row in rows], dtype=float)
        ax.plot(steps, success, color=SERIES_COLORS["real_ppo"], alpha=0.28, linewidth=1.2)
        per_seed_curves.append(np.interp(all_steps, steps, success))

    mean_curve = np.mean(per_seed_curves, axis=0)
    ax.plot(
        all_steps,
        mean_curve,
        color=SERIES_COLORS["real_ppo"],
        linewidth=2.2,
        label=f"{SERIES_LABELS['real_ppo']} (mean of {len(per_seed_curves)} seeds)",
    )

    crossover = {}
    if dream_level is not None:
        ax.axhline(
            dream_level,
            color=SERIES_COLORS["dream_ppo"],
            linewidth=2.2,
            label=f"{SERIES_LABELS['dream_ppo']} — 0 env steps",
        )
        ax.plot([0], [dream_level], "o", color=SERIES_COLORS["dream_ppo"], markersize=9, zorder=4)
        ax.annotate(
            f"dream-PPO {dream_level:.2f}\nat zero interaction",
            xy=(0, dream_level),
            xytext=(max(all_steps) * 0.04, dream_level + 0.09),
            color=SERIES_COLORS["dream_ppo"],
            fontsize=9,
            fontweight="bold",
        )

        above = mean_curve >= dream_level
        if above.any():
            first = int(np.argmax(above))
            # "Stays above" = no dip below the dream level from here to the end.
            sustained_index = next(
                (i for i in range(len(above)) if above[i:].all()), None
            )
            crossover = {
                "dream_level": dream_level,
                "first_match_env_steps": int(all_steps[first]),
                "sustained_match_env_steps": (
                    int(all_steps[sustained_index]) if sustained_index is not None else None
                ),
            }
            if sustained_index is not None:
                x = all_steps[sustained_index]
                ax.axvline(x, color=INK["muted"], linewidth=1.0, linestyle=(0, (4, 3)))
                ax.annotate(
                    f"real-PPO catches up:\n{x:,.0f} env steps",
                    xy=(x, dream_level),
                    xytext=(x * 1.04, 0.16),
                    color=INK["secondary"],
                    fontsize=9,
                )
        else:
            crossover = {"dream_level": dream_level, "first_match_env_steps": None}

    ax.set_xlabel("environment steps consumed")
    ax.set_ylabel("PushT success rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-max(all_steps) * 0.02)
    ax.xaxis.set_major_formatter(compact_steps_formatter())
    ax.set_title("What the world model is worth, in environment interaction", pad=12)
    ax.legend(loc="lower right")
    caption = "steps consumed = rollouts + real-env checkpoint selection"
    if len(per_seed_curves) > 1:
        caption = "thin lines = individual training seeds · " + caption
    fig.text(0.5, -0.04, caption, ha="center", color=INK["muted"], fontsize=8)
    save_figure(fig, output_root / "rq1_budget_curve.png")
    plt.close(fig)
    return crossover


def main() -> None:
    args = parse_args()
    apply_figure_style()
    input_root = repo_path(args.input_root)
    output_root = repo_path(args.output_root)

    results_path = input_root / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(
            f"{results_path} not found. Run `python -m scripts.rq1.evaluate_grid` first."
        )
    rows = read_jsonl(results_path)
    grouped = group_rows(rows)

    any_row = rows[0]
    args.repeats_note = (
        f"{any_row['eval']['repeats']} repeats x {any_row['eval']['episodes_per_repeat']} "
        f"episodes per checkpoint"
    )

    summary = write_table(grouped, output_root / "table.md", args)
    plot_success(grouped, args, output_root)

    curve_path = input_root / "budget_curve.jsonl"
    curve_rows = read_jsonl(curve_path) if curve_path.exists() else []
    crossover = plot_budget_curve(grouped, curve_rows, args, output_root)

    payload = {"table": summary, "crossover": crossover, "eval": any_row["eval"]}
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")

    if crossover.get("sustained_match_env_steps") is not None:
        print(
            f"\nHeadline: dream PPO reaches {crossover['dream_level']:.3f} at zero environment "
            f"interaction; real-env PPO needs ~{crossover['sustained_match_env_steps']:,} env "
            "steps to match and stay there."
        )


if __name__ == "__main__":
    main()
