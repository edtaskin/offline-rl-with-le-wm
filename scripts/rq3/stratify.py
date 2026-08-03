"""RQ3 arm (a): where does each agent's competence actually end?

A single pooled success rate hides the shape of the answer. PushT difficulty is
driven mainly by how far the block has to be transported, so this re-cuts
evaluations that have *already been run* by the initial block-to-goal distance,
and asks whether training inside the world model widens the region an agent can
handle or merely fills in the region the expert data already covered.

No policy is re-run. Every episode in an evaluation is seeded
(``config.seed + episode_index``) and ``reset(seed=...)`` reseeds the start-state
sampler, so replaying the resets alone recovers each episode's initial distance
exactly -- verified by construction in ``tests/test_rq3.py``. Those distances are
joined to the per-episode success already stored in ``metrics.json``.

Usage::

    python -m scripts.rq3.stratify
    python -m scripts.rq3.stratify --roots runs/evaluations runs/evaluations_rq3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from scripts.rq1.report import AXIS, COLORS, GRID, INK, INK_MUTED, wilson_interval

# The distance bins the analysis reports. Chosen to straddle the previously
# diagnosed cliff (competence collapsing somewhere past ~150 px) with enough
# resolution either side to see where it actually falls.
BIN_EDGES = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, float("inf"))

# Where dream episodes can start from: the anchor filter keeps expert states
# within this distance of the goal, so it is the edge of what imagination saw.
DREAM_ANCHOR_RADIUS = 200.0

RUN_NAME = re.compile(r"_rq\d_(?P<label>[a-z0-9-]+)_seed(?P<seed>[A-Za-z0-9]+)_(?P<ckpt>.+)$")

AGENT_ORDER = ("bc", "dream-ppo", "real-ppo")
AGENT_TITLES = {
    "bc": "Behavioural cloning",
    "dream-ppo": "PPO in imagination",
    "real-ppo": "PPO in the simulator",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", default=["runs/evaluations", "runs/evaluations_rq3"])
    parser.add_argument("--output-dir", default="runs/rq3")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["final", "bc"],
        help="only pool evaluations of these checkpoints (default: the RQ1 headline ones)",
    )
    parser.add_argument("--cache", default="runs/rq3/start_distances.json")
    return parser


# ------------------------------------------------------------------ discovery
def parse_run_dir(run_dir: Path) -> dict | None:
    match = RUN_NAME.search(run_dir.name)
    if not match:
        return None
    seed = match.group("seed")
    return {
        "label": match.group("label"),
        "seed": None if seed == "None" else int(seed),
        "checkpoint_name": match.group("ckpt"),
    }


def index_evaluations(roots, checkpoints) -> list[dict]:
    """Collect every finished evaluation, with its per-episode outcomes."""
    records = []
    for root in roots:
        for metrics_path in sorted(Path(root).glob("*/metrics.json")):
            meta = parse_run_dir(metrics_path.parent)
            if meta is None or meta["checkpoint_name"] not in checkpoints:
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = payload["config"]
            records.append(
                {
                    **meta,
                    "run_dir": str(metrics_path.parent),
                    "block_start_radius": config.get("block_start_radius"),
                    "max_episode_steps": config.get("max_episode_steps", 300),
                    "episodes": [
                        {"seed": episode["seed"], "success": float(episode["success"])}
                        for episode in payload["episodes"]
                    ],
                }
            )
    return records


# ------------------------------------------------------------------ distances
def compute_start_distances(radius, seeds, max_episode_steps) -> dict[int, float]:
    """Initial block-to-goal distance for each episode seed, by replaying resets.

    Reads the geometry off the environment rather than the reset ``info``: the
    ``block_goal_dist`` key only exists when the near-goal wrapper is active, and
    the unrestricted-start evaluations deliberately run without it.
    """
    from src.envs.pusht_wrappers import block_center, green_t_center
    from src.evaluation.pusht import PushTEvalConfig, make_evaluation_env

    config = PushTEvalConfig(
        episodes=1,
        seed=min(seeds),
        max_episode_steps=max_episode_steps,
        block_start_radius=radius,
    )
    env = make_evaluation_env(config)
    distances = {}
    try:
        for seed in sorted(seeds):
            env.reset(seed=int(seed))
            distances[int(seed)] = float(
                np.linalg.norm(block_center(env) - green_t_center(env))
            )
    finally:
        env.close()
    return distances


def load_start_distances(records, cache_path: Path) -> dict[tuple, dict[int, float]]:
    """Distances per (radius, max_episode_steps), cached across invocations."""
    cache = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        cache = {key: {int(s): d for s, d in value.items()} for key, value in raw.items()}

    needed = defaultdict(set)
    for record in records:
        key = f"{record['block_start_radius']}|{record['max_episode_steps']}"
        needed[key].update(episode["seed"] for episode in record["episodes"])

    changed = False
    for key, seeds in needed.items():
        known = cache.setdefault(key, {})
        missing = seeds - set(known)
        if not missing:
            continue
        radius_text, steps_text = key.split("|")
        radius = None if radius_text == "None" else float(radius_text)
        print(f"  replaying {len(missing)} resets for radius={radius_text} ...")
        known.update(compute_start_distances(radius, missing, int(steps_text)))
        changed = True

    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({k: {str(s): d for s, d in v.items()} for k, v in cache.items()}, indent=1),
            encoding="utf-8",
        )
    return cache


# ----------------------------------------------------------------- statistics
def stratify(records, distances) -> dict[str, list[dict]]:
    """Per agent, per distance bin: attempts and successes pooled over seeds."""
    tally = {label: [[0, 0.0] for _ in range(len(BIN_EDGES) - 1)] for label in AGENT_ORDER}
    for record in records:
        if record["label"] not in tally:
            continue
        key = f"{record['block_start_radius']}|{record['max_episode_steps']}"
        lookup = distances[key]
        for episode in record["episodes"]:
            distance = lookup[episode["seed"]]
            index = int(np.digitize(distance, BIN_EDGES) - 1)
            index = min(max(index, 0), len(BIN_EDGES) - 2)
            tally[record["label"]][index][0] += 1
            tally[record["label"]][index][1] += episode["success"]

    out = {}
    for label, bins in tally.items():
        rows = []
        for index, (attempts, successes) in enumerate(bins):
            low, high = wilson_interval(successes, attempts) if attempts else (np.nan, np.nan)
            rows.append(
                {
                    "lo": BIN_EDGES[index],
                    "hi": BIN_EDGES[index + 1],
                    "attempts": attempts,
                    "successes": successes,
                    "rate": successes / attempts if attempts else np.nan,
                    "ci": (low, high),
                }
            )
        out[label] = rows
    return out


def bin_label(row) -> str:
    return f"{row['lo']:.0f}+" if row["hi"] == float("inf") else f"{row['lo']:.0f}–{row['hi']:.0f}"


# -------------------------------------------------------------------- outputs
def plot(strata, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=300)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=11)

    centres = []
    for index in range(len(BIN_EDGES) - 1):
        low, high = BIN_EDGES[index], BIN_EDGES[index + 1]
        centres.append(low + 25.0 if high == float("inf") else (low + high) / 2)
    centres = np.array(centres)

    # The edge of the expert data, which is also the edge of what the imagined
    # trainer could sample a start state from.
    ax.axvspan(DREAM_ANCHOR_RADIUS, centres[-1] + 25, color=INK_MUTED, alpha=0.07, zorder=1)
    ax.annotate(
        "beyond where imagination\ncould start an episode",
        xy=(DREAM_ANCHOR_RADIUS + 12, 0.94), ha="left", va="top",
        fontsize=10.5, color=INK_MUTED, linespacing=1.4,
    )
    ax.axvline(DREAM_ANCHOR_RADIUS, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=2)

    for label in AGENT_ORDER:
        rows = strata.get(label)
        if not rows:
            continue
        mask = np.array([row["attempts"] > 0 for row in rows])
        if not mask.any():
            continue
        rates = np.array([row["rate"] for row in rows])
        lows = np.array([row["ci"][0] for row in rows])
        highs = np.array([row["ci"][1] for row in rows])
        ax.fill_between(
            centres[mask], lows[mask], highs[mask],
            color=COLORS[label], alpha=0.14, linewidth=0, zorder=3,
        )
        ax.plot(
            centres[mask], rates[mask], color=COLORS[label], linewidth=2.4, zorder=4,
            marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.2,
            label=AGENT_TITLES[label],
        )

    ax.set_xlabel("initial block→goal distance (pixels)", fontsize=12, color=INK)
    ax.set_ylabel("success rate", fontsize=12, color=INK)
    ax.set_xticks(centres)
    ax.set_xticklabels([bin_label(row) for row in strata[AGENT_ORDER[-1]]])
    ax.set_ylim(0, 1.02)
    ax.set_xlim(centres[0] - 25, centres[-1] + 25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(
        "Competence by how far the block has to travel",
        fontsize=17, color=INK, fontweight="bold", x=0.02, ha="left",
    )
    ax.set_title(
        "shaded bands are 95% confidence intervals · pooled over training seeds and start distributions",
        fontsize=10.5, color=INK_MUTED, loc="left", pad=10,
    )
    legend = ax.legend(loc="lower left", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_competence_by_distance.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {output_dir}/fig_competence_by_distance.png")


def write_table(strata, output_dir: Path) -> None:
    lines = [
        "# RQ3 (a) — competence stratified by initial block→goal distance",
        "",
        "| distance (px) | share of test episodes | "
        + " | ".join(AGENT_TITLES[a] for a in AGENT_ORDER)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in AGENT_ORDER) + " |",
    ]
    n_bins = len(BIN_EDGES) - 1
    # Share of the test set landing in each bin. Read off one agent because every
    # agent is evaluated on the same episode seeds; summing across agents would
    # just add three copies of the same start states together.
    reference = strata[AGENT_ORDER[-1]]
    total = sum(row["attempts"] for row in reference) or 1
    for index in range(n_bins):
        cells = []
        for label in AGENT_ORDER:
            row = strata[label][index]
            cells.append(f"{row['rate']:.0%} ({row['attempts']})" if row["attempts"] else "—")
        if not any(strata[label][index]["attempts"] for label in AGENT_ORDER):
            continue
        share = reference[index]["attempts"] / total
        lines.append(
            f"| {bin_label(reference[index])} | {share:.0%} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Cells are `success rate (episodes)`. Distances are each episode's true initial "
        "block→goal distance, recovered by replaying its seed.",
        "",
        "**Why the episode counts differ.** Across agents: behavioural cloning is one "
        "published checkpoint while each PPO agent has three training seeds, so the PPO "
        "columns carry exactly 3x as many episodes. Across rows: the initial distance is an "
        "outcome of the random start sampler, not a controlled variable, so the bins fill "
        "unevenly -- the `share of test episodes` column is that distribution.",
        "",
        f"Imagined training could only ever start an episode within {DREAM_ANCHOR_RADIUS:.0f} px "
        "of the goal, so every row past that is strictly outside what imagination saw.",
        "",
    ]
    path = output_dir / "table_competence_by_distance.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = index_evaluations(args.roots, set(args.checkpoints))
    if not records:
        raise SystemExit(f"no evaluations found under {args.roots}")
    counts = defaultdict(int)
    for record in records:
        counts[(record["label"], record["block_start_radius"])] += len(record["episodes"])
    print("RQ3 stratification | pooled episodes:")
    # Sort with None (unrestricted starts) last rather than comparing it to a float.
    for (label, radius), total in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1] or 0)):
        print(f"  {label:11s} radius={str(radius):>6s}  {total:5d} episodes")

    distances = load_start_distances(records, Path(args.cache))
    strata = stratify(records, distances)
    write_table(strata, output_dir)
    plot(strata, output_dir)


if __name__ == "__main__":
    main()
