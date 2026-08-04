"""Shared plumbing for the RQ1 / RQ2 poster experiments.

Both research questions compare the same agents on the same task, so anything
that must not silently drift between them lives here: how a checkpoint becomes a
success rate (always the canonical evaluator in :mod:`src.evaluation`), how a
training run directory is located, how success rates get error bars, and the
figure palette.

Nothing in this module trains or evaluates on its own -- it is imported by the
scripts under ``scripts/rq1`` and ``scripts/rq2``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


# --------------------------------------------------------------------- style
# Categorical slots 1-3 of the validated default palette, in fixed order. Three
# slots is the documented all-pairs cap, which is exactly what we need: colour
# tracks the *entity* (dream / real / BC), never its rank, so "dream" is the same
# blue in an RQ1 budget curve and an RQ2 optimism plot.
SERIES_COLORS = {
    "dream_ppo": "#2a78d6",  # slot 1, blue
    "real_ppo": "#eb6834",  # slot 2, orange
    "bc": "#1baf7a",  # slot 3, aqua
}
# RQ2 plots the two *measurements* of one agent, so they inherit the agent-level
# meaning: imagined = the dream world's colour, real = the simulator's.
SERIES_COLORS["dream"] = SERIES_COLORS["dream_ppo"]
SERIES_COLORS["real"] = SERIES_COLORS["real_ppo"]

SERIES_LABELS = {
    "dream_ppo": "dream-PPO",
    "real_ppo": "real-PPO",
    "bc": "BC",
    "dream": "imagined (dream)",
    "real": "real held-out",
}

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
}


def apply_figure_style() -> None:
    """Recessive chrome, thin marks, no chartjunk -- applied once per script."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "axes.edgecolor": INK["muted"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "text.color": INK["primary"],
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "font.size": 10,
            "figure.dpi": 160,
        }
    )


def compact_steps_formatter():
    """Tick formatter rendering step counts as 400k / 1.2M.

    Matplotlib's default offset notation puts a stray ``1e6`` in the corner,
    where it collides with the axis label and reads as chart noise.
    """
    from matplotlib.ticker import FuncFormatter

    def format_tick(value, _position):
        value = float(value)
        if abs(value) >= 1e6:
            return f"{value / 1e6:g}M"
        if abs(value) >= 1e3:
            return f"{value / 1e3:g}k"
        return f"{value:g}"

    return FuncFormatter(format_tick)


def save_figure(fig, path: str | Path) -> Path:
    """Write a figure as both PNG (slides) and PDF (poster print)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {path} and {pdf_path}")
    return path


# ----------------------------------------------------------------- run lookup
@dataclass(frozen=True)
class RunDir:
    """One training run's output directory, with the seed it was trained at."""

    exp_name: str
    seed: int
    path: Path

    @property
    def selection_log(self) -> Path:
        return self.path / "selection_log.jsonl"


def _run_stamp_key(name: str) -> tuple:
    """Sort key for a ``%d%m%Y-%H%M%S`` run stamp.

    The stamp is day-first, so lexicographic ordering picks the wrong run as
    soon as two runs land in different months; parse it properly and fall back
    to the raw name for anything that is not a stamp.
    """
    try:
        return (1, datetime.strptime(name, "%d%m%Y-%H%M%S").timestamp())
    except ValueError:
        return (0, 0.0)


def find_run_dir(
    exp_name: str,
    seed: int,
    runs_root: str | Path = "runs",
    timestamp: str | None = None,
) -> RunDir:
    """Locate ``<runs_root>/<exp_name>__seed<seed>/<stamp>``, newest stamp by default."""
    parent = repo_path(runs_root) / f"{exp_name}__seed{seed}"
    if not parent.is_dir():
        raise FileNotFoundError(
            f"No run directory for exp={exp_name!r} seed={seed} under {repo_path(runs_root)}. "
            "Train it first (see scripts/rq1/run_campaign.sh)."
        )
    if timestamp is not None:
        run = parent / timestamp
        if not run.is_dir():
            raise FileNotFoundError(f"No such run directory: {run}")
        return RunDir(exp_name, seed, run)

    stamps = [child for child in parent.iterdir() if child.is_dir()]
    if not stamps:
        raise FileNotFoundError(f"{parent} contains no run directories")
    newest = max(stamps, key=lambda child: _run_stamp_key(child.name))
    return RunDir(exp_name, seed, newest)


def snapshot_checkpoints(run_dir: Path) -> list[dict]:
    """Every ``snapshot_step<budget>_it<iteration>.pt`` in a run, iteration-ordered.

    The env-step budget is encoded in the filename by ``save_snapshot`` precisely
    so a post-hoc interaction-budget curve is possible; the iteration lets a
    snapshot be paired with the matching ``selection_log.jsonl`` row.
    """
    snapshots = []
    for path in sorted(Path(run_dir).glob("snapshot_step*_it*.pt")):
        stem = path.stem  # snapshot_step000004405_it00010
        try:
            step_part, iter_part = stem.split("_it")
            env_steps = int(step_part.removeprefix("snapshot_step"))
            iteration = int(iter_part)
        except ValueError:
            continue
        snapshots.append({"path": path, "env_steps": env_steps, "iteration": iteration})
    return sorted(snapshots, key=lambda entry: entry["iteration"])


def checkpoint_budget(path: str | Path) -> dict:
    """Interaction budget and metadata recorded inside a PPO checkpoint.

    ``train_env_steps`` is the rollout cost and ``eval_env_steps`` the cost of
    real-env checkpoint selection; RQ1 reports their sum, because selecting on
    the simulator is environment interaction even though it trains nothing.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    return {
        "env_steps_consumed": int(payload.get("env_steps_consumed", 0)),
        "train_env_steps": int(payload.get("train_env_steps", payload.get("global_step", 0))),
        "eval_env_steps": int(payload.get("eval_env_steps", 0)),
        "imagined_steps": int(payload.get("global_step", 0)),
        "recorded_success_rate": payload.get("success_rate"),
        "selection": config.get("selection"),
        "exp_name": config.get("exp_name"),
        "seed": config.get("seed"),
    }


def read_selection_log(run_dir: Path) -> list[dict]:
    """Rows of ``selection_log.jsonl`` (dream trainer only), iteration-ordered.

    The last iteration appears twice -- once from the in-loop selection pass and
    once from ``_finalize_selection`` -- so duplicates are collapsed keeping the
    final (later) measurement.
    """
    path = Path(run_dir) / "selection_log.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Only src.ppo.train_lewm writes a selection log; "
            "make sure this is a dream-PPO run."
        )
    by_iteration: dict[int, dict] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_iteration[int(row["iteration"])] = row
    return [by_iteration[key] for key in sorted(by_iteration)]


# ------------------------------------------------------------------- evaluation
CANONICAL_EVAL = {
    "episodes": 50,
    "repeats": 3,
    "seed": 42,
    "max_episode_steps": 300,
    # Training used --block_start_near_goal --block_start_radius 200, so
    # evaluation has to sample starts from the same distribution.
    "block_start_radius": 200.0,
}


def canonical_eval_argv(
    agent_type: str,
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    run_name: str,
    stats: str | None = None,
    device: str = "auto",
    episodes: int = CANONICAL_EVAL["episodes"],
    repeats: int = CANONICAL_EVAL["repeats"],
    seed: int = CANONICAL_EVAL["seed"],
    block_start_radius: float = CANONICAL_EVAL["block_start_radius"],
    max_episode_steps: int = CANONICAL_EVAL["max_episode_steps"],
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Argument vector for ``python -m src.evaluation.evaluate_pusht``.

    Built as a literal argv (rather than a hand-made namespace) so that what the
    poster reports and what the documented command does cannot come apart.
    """
    argv = [
        "--agent-type", agent_type,
        "--checkpoint", str(checkpoint),
        "--episodes", str(episodes),
        "--repeats", str(repeats),
        "--seed", str(seed),
        "--max-episode-steps", str(max_episode_steps),
        "--block-start-radius", str(block_start_radius),
        "--output-root", str(output_root),
        "--run-name", run_name,
        "--device", device,
    ]
    if stats:
        argv += ["--stats", str(stats)]
    return argv + list(extra)


def run_canonical_eval(*args, **kwargs):
    """Run the canonical evaluator end to end and return its aggregate result."""
    from src.evaluation.evaluate_pusht import build_parser, evaluate_from_args

    argv = canonical_eval_argv(*args, **kwargs)
    print("$ python -m src.evaluation.evaluate_pusht " + " ".join(argv))
    return evaluate_from_args(build_parser().parse_args(argv))


def evaluate_agent_repeats(
    agent,
    *,
    episodes: int = CANONICAL_EVAL["episodes"],
    repeats: int = CANONICAL_EVAL["repeats"],
    seed: int = CANONICAL_EVAL["seed"],
    seed_stride: int = 1,
    block_start_radius: float = CANONICAL_EVAL["block_start_radius"],
    max_episode_steps: int = CANONICAL_EVAL["max_episode_steps"],
):
    """The canonical repeat loop for an already-constructed evaluation agent.

    Identical to what :func:`run_canonical_eval` does -- same
    :class:`PushTEvalConfig`, same non-overlapping seed ranges, a fresh env per
    repeat -- minus the argparse and run-directory shell. Used where one encoder
    is amortized over many checkpoints (the budget curve), which is the only
    thing that makes evaluating every snapshot affordable.
    """
    from src.evaluation.evaluate_pusht import make_repeat_seeds
    from src.evaluation.pusht import (
        PushTEvalConfig,
        aggregate_evaluation_results,
        run_evaluation,
    )

    results = []
    for repeat_seed in make_repeat_seeds(seed, repeats, episodes, seed_stride):
        config = PushTEvalConfig(
            episodes=episodes,
            seed=repeat_seed,
            seed_stride=seed_stride,
            max_episode_steps=max_episode_steps,
            block_start_radius=block_start_radius,
        )
        results.append(run_evaluation(agent, config))
    return aggregate_evaluation_results(results)


# ------------------------------------------------------------------ statistics
def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate.

    Preferred over the normal approximation because success rates here sit near
    0.9 with n in the low hundreds, where the naive interval overshoots 1.0.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- the ranking Spearman's rho is defined on."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    """Rank correlation, implemented locally to avoid a scipy dependency."""
    return pearson(_rank(x), _rank(y))


# ----------------------------------------------------------------------- io
def write_jsonl(path: str | Path, rows) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def append_jsonl(path: str | Path, row: dict) -> None:
    """Append one row immediately, so a long campaign survives an interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def upsert_jsonl(path: str | Path, row: dict, key_fields: tuple[str, ...]) -> None:
    """Append ``row``, replacing any existing row with the same key.

    Results files are written incrementally so an interrupted campaign can be
    resumed, but a re-evaluation (``--overwrite``) must *replace* its old row
    rather than sit beside it -- two rows for one checkpoint would be silently
    averaged together by the report.
    """
    path = Path(path)
    key = tuple(row.get(field) for field in key_fields)
    kept = [
        existing
        for existing in (read_jsonl(path) if path.exists() else [])
        if tuple(existing.get(field) for field in key_fields) != key
    ]
    write_jsonl(path, kept + [row])


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def seed_list(raw: list[str] | list[int]) -> list[int]:
    return [int(value) for value in raw]
