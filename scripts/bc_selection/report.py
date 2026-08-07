"""Aggregate the BC-checkpoint sweep and recommend a checkpoint to freeze.

The recommendation is deliberately NOT the argmax. With three seeds per
checkpoint the across-seed spread is comparable to the differences being ranked,
so the argmax is substantially a draw for the luckiest seed and does not
reproduce. Instead:

* Within-checkpoint variance is **pooled across checkpoints**. Three seeds give a
  useless per-cell variance estimate; assuming one common seed-noise scale (which
  is what it is -- PPO RNG, not something the BC changes) turns 3 samples into
  ``3 x n_checkpoints`` and makes the error bars mean something.
* Every checkpoint statistically indistinguishable from the top one forms the
  **plateau**, and the recommendation is the middle of the longest contiguous run
  of it. A checkpoint that only wins by less than the seed spread is not a
  finding; the middle of a plateau is the choice that survives re-running.

Also reported: the rank correlation between a checkpoint's standalone BC success
and the PPO success it initializes. A weak or negative correlation is the
interesting outcome -- it says the best-fit BC is not the best RL initialization,
which is the mechanism that justifies having run the sweep at all.

The decision is recorded in three places, all written by this script: a manifest
JSON next to the results, a W&B run (``--wandb``) carrying the full grid as
tables, and -- once the choice is final -- the checkpoint itself on the Hugging
Face Hub (``--push-to-hf``), so later experiments can pin one ``hf://`` URI
instead of a path inside somebody's ``runs/``.

Usage::

    python -m scripts.bc_selection.report
    python -m scripts.bc_selection.report --variant final --plateau-z 1.0
    python -m scripts.bc_selection.report --wandb
    python -m scripts.bc_selection.report --wandb --push-to-hf
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.rq_common import (
    CANONICAL_EVAL,
    INK,
    SERIES_COLORS,
    SERIES_LABELS,
    apply_figure_style,
    read_jsonl,
    repo_path,
    save_figure,
    spearman,
    wilson_interval,
)
from src.bc.tracking import args_for_config, init_wandb, json_safe, log_wandb_artifact
from src.utils.hf_hub import push_files_to_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="runs/bc_selection/results.jsonl")
    parser.add_argument("--output-root", default="runs/bc_selection")
    parser.add_argument("--variant", default="best", choices=["best", "final"])
    parser.add_argument(
        "--plateau-z",
        type=float,
        default=1.0,
        help="Plateau = every checkpoint within z standard errors of the top mean",
    )
    parser.add_argument("--no-figure", action="store_true")

    # ------------------------------------------------------------------ W&B
    # Names match src/bc/tracking.py's init_wandb so it can consume this
    # namespace directly, the same way train_bc_latent.py does.
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the sweep grid, the recommendation and the figure to Weights & Biases",
    )
    parser.add_argument("--wandb_project", default="offline-rl-lewm")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default="bc_selection")
    parser.add_argument("--wandb_tags", nargs="*", default=["bc_selection"])
    parser.add_argument("--wandb_mode", default=None, choices=[None, "online", "offline", "disabled"])

    # ------------------------------------------------- Hugging Face publishing
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Upload the recommended checkpoint, its stats sidecar and the manifest to the Hub",
    )
    parser.add_argument("--hf-repo-id", default="offline-rl-with-le-wm/bc")
    parser.add_argument(
        "--hf-path-prefix",
        default=None,
        help="Directory inside the repo; defaults to a dated, checkpoint-specific name so a "
        "publish can never land on a path that is already referenced",
    )
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument(
        "--hf-allow-overwrite",
        action="store_true",
        help="Permit writing over files that already exist at the target path (see _assert_free)",
    )
    parser.add_argument(
        "--bc-stats",
        default=None,
        help="Stats sidecar to publish beside the checkpoint; defaults to the single "
        "*_stats.pth next to it",
    )
    parser.add_argument(
        "--allow-unconfirmed",
        action="store_true",
        help="Publish a recommendation that has no confirmation seeds (not advised: the "
        "selection mean is optimistically biased)",
    )
    return parser.parse_args()


class Cell:
    """One BC checkpoint's PPO results across seeds."""

    def __init__(self, tag: str, epoch: int | None):
        self.tag = tag
        self.epoch = epoch
        self.select: list[tuple[int, float]] = []   # (seed, success)
        self.confirm: list[tuple[int, float]] = []
        self.bc_standalone: float | None = None
        self.bc_weight_l2: float | None = None
        self.episodes: int = 0
        # Path the sweep ran this checkpoint from; the publish step needs it to
        # find the file it is being asked to upload.
        self.bc_checkpoint: str | None = None

    @property
    def values(self) -> np.ndarray:
        return np.array([success for _, success in self.select], dtype=float)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if len(self.values) else float("nan")

    @property
    def n(self) -> int:
        return len(self.select)

    def variance(self) -> float | None:
        """Unbiased across-seed variance; ``None`` when a single seed ran."""
        if self.n < 2:
            return None
        return float(self.values.var(ddof=1))


def load_cells(rows: list[dict], variant: str) -> list[Cell]:
    cells: dict[str, Cell] = {}

    def cell_for(row) -> Cell:
        tag = row["bc_tag"]
        if tag not in cells:
            cells[tag] = Cell(tag, row.get("bc_epoch"))
        cell = cells[tag]
        if cell.bc_checkpoint is None:
            cell.bc_checkpoint = row.get("bc_checkpoint")
        return cell

    for row in rows:
        if row.get("method") == "bc" and row.get("variant") == "standalone":
            cell = cell_for(row)
            cell.bc_standalone = row["success_rate"]
            cell.bc_weight_l2 = row.get("bc_weight_l2")
        elif row.get("method") == "dream_ppo" and row.get("variant") == variant:
            cell = cell_for(row)
            entry = (row["seed"], row["success_rate"])
            if row.get("seed_role") == "confirm":
                cell.confirm.append(entry)
            else:
                cell.select.append(entry)
            cell.episodes = row.get("episodes", cell.episodes)

    ordered = list(cells.values())
    # Numeric epoch order; the run's final (unsuffixed) checkpoint has no epoch
    # number and is placed after every numbered one.
    known = [cell.epoch for cell in ordered if cell.epoch is not None]
    fallback = (max(known) + 1) if known else 0
    ordered.sort(key=lambda cell: (cell.epoch if cell.epoch is not None else fallback, cell.tag))
    return ordered


def pooled_sd(cells: list[Cell]) -> tuple[float, int]:
    """Pooled within-checkpoint SD across seeds, and its degrees of freedom."""
    numerator = 0.0
    dof = 0
    for cell in cells:
        variance = cell.variance()
        if variance is not None:
            numerator += (cell.n - 1) * variance
            dof += cell.n - 1
    if dof == 0:
        return float("nan"), 0
    return math.sqrt(numerator / dof), dof


def find_plateau(cells: list[Cell], s_pooled: float, z: float) -> tuple[list[Cell], Cell]:
    """Checkpoints indistinguishable from the top, and the one to recommend.

    The threshold is the standard error of a *difference* between two means, not
    of one mean: the comparison being made is "is this checkpoint worse than the
    best one", which involves the noise in both.
    """
    scored = [cell for cell in cells if cell.n > 0]
    top = max(scored, key=lambda cell: cell.mean)
    if not math.isfinite(s_pooled):
        return [top], top

    plateau = []
    for cell in scored:
        se_difference = s_pooled * math.sqrt(1.0 / cell.n + 1.0 / top.n)
        if cell.mean >= top.mean - z * se_difference:
            plateau.append(cell)

    # Longest contiguous run (in epoch order) containing the top checkpoint: an
    # isolated checkpoint that happens to clear the threshold is noise, whereas a
    # run of neighbours that all clear it is a real region of the curve.
    order = {cell.tag: index for index, cell in enumerate(scored)}
    in_plateau = {cell.tag for cell in plateau}
    best_run: list[Cell] = []
    current: list[Cell] = []
    for cell in scored:
        if cell.tag in in_plateau:
            current.append(cell)
        else:
            current = []
        if current and order[top.tag] >= order[current[0].tag]:
            if order[top.tag] <= order[current[-1].tag] and len(current) > len(best_run):
                best_run = list(current)
    run = best_run or [top]
    # Lower-middle on an even-length plateau: among checkpoints that are
    # statistically tied, the earlier one is the cheaper and less overfit choice.
    return run, run[(len(run) - 1) // 2]


def print_table(cells: list[Cell], s_pooled: float, dof: int, plateau, recommended) -> None:
    plateau_tags = {cell.tag for cell in plateau}
    header = (
        f"{'BC checkpoint':<28} {'epoch':>6} {'n':>3} {'PPO mean':>9} {'sd':>7} "
        f"{'range':>15} {'BC solo':>8}  flag"
    )
    print(header)
    print("-" * len(header))
    for cell in cells:
        if cell.n == 0:
            print(f"{cell.tag:<28} {str(cell.epoch or '-'):>6} {'0':>3}  (no PPO runs)")
            continue
        variance = cell.variance()
        sd = f"{math.sqrt(variance):.3f}" if variance is not None else "-"
        low, high = cell.values.min(), cell.values.max()
        solo = f"{cell.bc_standalone:.3f}" if cell.bc_standalone is not None else "-"
        flags = []
        if cell.tag in plateau_tags:
            flags.append("plateau")
        if cell is recommended:
            flags.append("<- RECOMMENDED")
        print(
            f"{cell.tag:<28} {str(cell.epoch if cell.epoch is not None else 'final'):>6} "
            f"{cell.n:>3} {cell.mean:>9.3f} {sd:>7} "
            f"{low:>7.3f}-{high:<7.3f} {solo:>8}  {' '.join(flags)}"
        )
    print()
    print(f"pooled across-seed SD: {s_pooled:.4f} (dof {dof})")


def main() -> None:
    args = parse_args()
    results_path = repo_path(args.results)
    if not results_path.exists():
        raise SystemExit(
            f"{results_path} not found. Run scripts/bc_selection/evaluate_sweep.py first."
        )
    cells = load_cells(read_jsonl(results_path), args.variant)
    scored = [cell for cell in cells if cell.n > 0]
    if not scored:
        raise SystemExit(f"No dream_ppo rows for variant={args.variant!r} in {results_path}.")

    s_pooled, dof = pooled_sd(cells)
    plateau, recommended = find_plateau(cells, s_pooled, args.plateau_z)

    print(f"\nBC-checkpoint sweep | variant={args.variant} | {len(scored)} checkpoints\n")
    print_table(cells, s_pooled, dof, plateau, recommended)

    single_seed = [cell.tag for cell in scored if cell.n < 2]
    if dof == 0:
        print(
            "\nWARNING: no checkpoint has >=2 seeds, so seed noise is unmeasured and the "
            "'recommendation' is a bare argmax. Re-run the sweep with --seeds \"1 2 3\"."
        )
    elif single_seed:
        print(
            f"\nWARNING: {len(single_seed)} checkpoint(s) have <2 seeds "
            f"({', '.join(single_seed)}). Their means are single draws; the pooled SD "
            "above is estimated from the other checkpoints and applied to them anyway."
        )

    # ------------------------------------------------------------- mechanism
    rho = rho_weight = None
    paired = [
        (cell.bc_standalone, cell.mean) for cell in scored if cell.bc_standalone is not None
    ]
    if len(paired) >= 3:
        rho = spearman([a for a, _ in paired], [b for _, b in paired])
        print(f"\nSpearman(BC standalone success, PPO success) = {rho:+.3f}  (n={len(paired)})")
        if rho < 0.3:
            print(
                "  -> A better-fitting BC is not a better RL initialization here. That is a "
                "reportable finding, not a nuisance: state it rather than just the epoch."
            )

    weights = [(cell.bc_weight_l2, cell.mean) for cell in scored if cell.bc_weight_l2]
    if len(weights) >= 3:
        rho_weight = spearman([a for a, _ in weights], [b for _, b in weights])
        print(f"Spearman(BC weight L2, PPO success)          = {rho_weight:+.3f}  (n={len(weights)})")

    # ------------------------------------------------------------ the verdict
    print(f"\nPlateau ({args.plateau_z}-SE of the top mean): {', '.join(c.tag for c in plateau)}")
    plural = "" if recommended.n == 1 else "s"
    print(
        f"RECOMMENDED: {recommended.tag}  "
        f"(mean {recommended.mean:.3f} over {recommended.n} seed{plural})"
    )

    if recommended.episodes:
        low, high = wilson_interval(recommended.mean * recommended.episodes, recommended.episodes)
        # Episode-sampling noise for ONE run sitting at this mean -- NOT a
        # confidence interval on the across-seed mean printed above. Seed noise
        # is the larger term here (pooled SD above); quote that one in the
        # writeup and this only to say how much 150 episodes resolves.
        print(
            f"  episode-sampling 95% CI for a single run at this mean "
            f"({recommended.episodes} episodes): [{low:.3f}, {high:.3f}]"
        )

    if recommended.confirm:
        values = np.array([success for _, success in recommended.confirm])
        print(
            f"  CONFIRMATION on {len(values)} fresh seeds: mean {values.mean():.3f} "
            f"(range {values.min():.3f}-{values.max():.3f})"
        )
        print("  Report this number, not the selection mean above.")
    else:
        seeds = " ".join(str(seed + 10) for seed, _ in recommended.select)
        checkpoint = Path(recommended.bc_checkpoint) if recommended.bc_checkpoint else None
        stats = _sidecar(checkpoint, "_stats.pth") if checkpoint else None
        print(
            "  NOT YET CONFIRMED. The selection mean is optimistically biased -- it is the "
            "max over checkpoints of a noisy quantity. Re-run the winner on fresh seeds:\n"
            f"    bash scripts/bc_selection/run_sweep.sh \\\n"
            f"      --bc-checkpoints {checkpoint or '<dir>/' + recommended.tag + '.pth'} \\\n"
            f"      --bc-stats {stats or '<dir>/<run>_stats.pth'} \\\n"
            # --seed-role confirm is not optional: without it the sweep files
            # these seeds as selection seeds and the confirmation is lost.
            f"      --seeds \"{seeds}\" --seed-role confirm\n"
            "  That re-runs the evaluation and this report for you; publish afterwards."
        )

    print(
        "\nBefore reusing this checkpoint across reward probes, spot-check that the ranking "
        "holds: re-run the recommendation and ONE neighbour under a different probe. If the "
        "order flips, the BC cannot be frozen across probes -- and that is itself the result."
    )

    figure_path = None
    if not args.no_figure:
        figure_path = make_figure(cells, plateau, recommended, args)

    # --------------------------------------------------------- record keeping
    summary = build_summary(
        args, scored, s_pooled, dof, plateau, recommended, rho, rho_weight, results_path
    )
    manifest_path = repo_path(args.output_root) / "selection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(f"\nmanifest -> {manifest_path}")

    # Publishing runs before the W&B log so the resulting hf:// URI is part of
    # the same record that justifies it. Nothing is lost if it raises: the
    # expensive artefact, results.jsonl, is already on disk.
    upload = None
    if args.push_to_hf:
        upload = publish_to_hub(args, recommended, summary, manifest_path, figure_path)
        summary["hf"] = {
            "repo_id": upload.repo_id,
            "repo_url": upload.repo_url,
            "uploaded_files": list(upload.uploaded_files),
            "checkpoint_uri": summary["hf_checkpoint_uri"],
            "stats_uri": summary["hf_stats_uri"],
        }
        manifest_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")

    if args.wandb:
        log_sweep_to_wandb(args, scored, summary, results_path, manifest_path, figure_path)


# --------------------------------------------------------------------- summary
def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def build_summary(
    args, scored, s_pooled, dof, plateau, recommended, rho, rho_weight, results_path
) -> dict:
    """Everything needed to defend the choice, in one serialisable dict.

    The same dict backs the local manifest, the W&B summary and the JSON that
    travels to the Hub beside the checkpoint, so those three cannot disagree.
    """
    confirm = np.array([success for _, success in recommended.confirm], dtype=float)
    variance = recommended.variance()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "results_path": str(results_path),
        "recommended": {
            "bc_tag": recommended.tag,
            "bc_epoch": recommended.epoch,
            "bc_checkpoint": recommended.bc_checkpoint,
            "select_mean": recommended.mean,
            "select_sd": math.sqrt(variance) if variance is not None else None,
            "select_seeds": [seed for seed, _ in recommended.select],
            "bc_standalone_success": recommended.bc_standalone,
            "bc_weight_l2": recommended.bc_weight_l2,
        },
        "confirmation": {
            "seeds": [seed for seed, _ in recommended.confirm],
            "mean": float(confirm.mean()),
            "min": float(confirm.min()),
            "max": float(confirm.max()),
        }
        if len(confirm)
        else None,
        "selection_rule": {
            "variant": args.variant,
            "plateau_z": args.plateau_z,
            "plateau": [cell.tag for cell in plateau],
            "pooled_seed_sd": s_pooled if math.isfinite(s_pooled) else None,
            "pooled_dof": dof,
            "rule": "middle of the longest contiguous plateau, not the argmax",
        },
        "mechanism": {
            "spearman_bc_standalone_vs_ppo": rho,
            "spearman_bc_weight_l2_vs_ppo": rho_weight,
        },
        "evaluation": {
            "episodes": recommended.episodes or CANONICAL_EVAL["episodes"],
            "protocol": "scripts.rq_common.CANONICAL_EVAL",
        },
        "checkpoints": [
            {
                "bc_tag": cell.tag,
                "bc_epoch": cell.epoch,
                "n_select": cell.n,
                "ppo_mean": cell.mean,
                "bc_standalone_success": cell.bc_standalone,
            }
            for cell in scored
        ],
    }


# --------------------------------------------------------------------- W&B
def _wandb_tables(scored, recommended, plateau):
    import wandb

    plateau_tags = {cell.tag for cell in plateau}
    per_checkpoint = wandb.Table(
        columns=[
            "bc_tag", "bc_epoch", "n_select", "ppo_mean", "ppo_sd", "ppo_min", "ppo_max",
            "bc_standalone", "bc_weight_l2", "plateau", "recommended",
        ]
    )
    # One row per PPO run as well as the aggregate: the spread is the whole
    # reason the recommendation is not an argmax, and a table of means alone
    # hides it.
    per_seed = wandb.Table(columns=["bc_tag", "bc_epoch", "seed", "seed_role", "success_rate"])
    for cell in scored:
        variance = cell.variance()
        per_checkpoint.add_data(
            cell.tag,
            cell.epoch,
            cell.n,
            cell.mean,
            math.sqrt(variance) if variance is not None else None,
            float(cell.values.min()),
            float(cell.values.max()),
            cell.bc_standalone,
            cell.bc_weight_l2,
            cell.tag in plateau_tags,
            cell is recommended,
        )
        for seed, success in cell.select:
            per_seed.add_data(cell.tag, cell.epoch, seed, "select", success)
        for seed, success in cell.confirm:
            per_seed.add_data(cell.tag, cell.epoch, seed, "confirm", success)
    return per_checkpoint, per_seed


def log_sweep_to_wandb(args, scored, summary, results_path, manifest_path, figure_path) -> None:
    """One W&B run per report: the grid as tables, the verdict as summary keys."""
    import wandb

    run = init_wandb(args, {**args_for_config(args), "results_path": str(results_path)})
    if run is None:
        return

    plateau_tags = set(summary["selection_rule"]["plateau"])
    plateau = [cell for cell in scored if cell.tag in plateau_tags]
    recommended = next(
        cell for cell in scored if cell.tag == summary["recommended"]["bc_tag"]
    )
    per_checkpoint, per_seed = _wandb_tables(scored, recommended, plateau)

    payload = {"bc_selection/per_checkpoint": per_checkpoint, "bc_selection/per_seed": per_seed}
    if figure_path is not None and Path(figure_path).exists():
        payload["bc_selection/figure"] = wandb.Image(str(figure_path))
    run.log(payload)

    # Flattened onto summary so a W&B table view can sort sweeps against each
    # other; nested dicts do not surface as sortable columns.
    run.summary.update(
        json_safe(
            {
                "recommended_bc_tag": summary["recommended"]["bc_tag"],
                "recommended_bc_epoch": summary["recommended"]["bc_epoch"],
                "recommended_select_mean": summary["recommended"]["select_mean"],
                "recommended_select_sd": summary["recommended"]["select_sd"],
                "recommended_bc_standalone": summary["recommended"]["bc_standalone_success"],
                "confirmed": summary["confirmation"] is not None,
                "confirmation_mean": (summary["confirmation"] or {}).get("mean"),
                "pooled_seed_sd": summary["selection_rule"]["pooled_seed_sd"],
                "pooled_dof": summary["selection_rule"]["pooled_dof"],
                "plateau_size": len(summary["selection_rule"]["plateau"]),
                "plateau": ", ".join(summary["selection_rule"]["plateau"]),
                "variant": args.variant,
                "num_checkpoints": len(scored),
                "spearman_bc_vs_ppo": summary["mechanism"]["spearman_bc_standalone_vs_ppo"],
                "spearman_weight_vs_ppo": summary["mechanism"]["spearman_bc_weight_l2_vs_ppo"],
                "hf_checkpoint_uri": summary.get("hf_checkpoint_uri"),
                "hf_repo_url": (summary.get("hf") or {}).get("repo_url"),
            }
        )
    )

    log_wandb_artifact(
        run,
        artifact_name="bc-selection-sweep",
        artifact_type="evaluation",
        file_paths=[
            str(results_path),
            str(manifest_path),
            str(figure_path) if figure_path else None,
        ],
        metadata=summary,
    )
    print(f"W&B run: {run.url}")
    run.finish()


# ------------------------------------------------------------ Hugging Face
def _sidecar(checkpoint: Path, suffix: str, override: str | None = None) -> Path | None:
    """Locate a per-run sidecar next to a per-epoch checkpoint.

    The sidecar is written once per BC *run* under the base checkpoint's name
    (``pusht_bc_stats.pth``), while the sweep selects a per-epoch file
    (``pusht_bc_epoch60.pth``), so the name cannot be derived from the selected
    checkpoint and the directory is globbed instead.
    """
    if override:
        return repo_path(override)
    matches = sorted(checkpoint.parent.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _assert_free(repo_id, path_prefix, filenames, token, allow_overwrite) -> None:
    """Refuse to publish onto paths that already hold files.

    ``hf://`` references cannot pin a revision (see the note in README's artifact
    section), so every consumer of a path always resolves to whatever is at HEAD.
    Overwriting a published file therefore does not create a new version, it
    silently redefines every existing reference to the old one.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    prefix = (path_prefix or "").strip("/")
    targets = {f"{prefix}/{name}" if prefix else name for name in filenames}
    try:
        existing = set(HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model"))
    except RepositoryNotFoundError:
        return  # new repo; nothing to clobber
    collisions = sorted(targets & existing)
    if collisions and not allow_overwrite:
        raise SystemExit(
            f"Refusing to publish: {len(collisions)} file(s) already exist at "
            f"{repo_id}/{prefix}:\n  " + "\n  ".join(collisions) + "\n"
            "hf:// references cannot pin a revision, so overwriting these would silently "
            "redefine every existing reference to them. Publish under a new "
            "--hf-path-prefix, or pass --hf-allow-overwrite if that is genuinely intended."
        )


def publish_to_hub(args, recommended, summary, manifest_path, figure_path):
    """Upload the selected checkpoint and its provenance to the Hub."""
    if summary["confirmation"] is None and not args.allow_unconfirmed:
        raise SystemExit(
            "Refusing to publish an unconfirmed recommendation. The selection mean is the max "
            "over checkpoints of a noisy quantity and is optimistically biased by roughly the "
            "effect being claimed. Re-run the winner on fresh seeds (see the command printed "
            "above), then re-run this report -- or pass --allow-unconfirmed to publish anyway."
        )
    if not recommended.bc_checkpoint:
        raise SystemExit(
            f"No bc_checkpoint recorded for {recommended.tag} in results.jsonl; cannot publish. "
            "Re-run evaluate_sweep.py to regenerate the rows."
        )
    checkpoint = repo_path(recommended.bc_checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Selected checkpoint no longer on disk: {checkpoint}")

    stats = _sidecar(checkpoint, "_stats.pth", args.bc_stats)
    if stats is None or not stats.is_file():
        raise SystemExit(
            f"No unique *_stats.pth beside {checkpoint}. Pass --bc-stats explicitly: PPO reads "
            "the agent contract (frame_stack / frame_stride / action_chunk_size) off it, so a "
            "checkpoint published without its stats file is unusable."
        )
    run_config = _sidecar(checkpoint, "_run_config.json")

    # Dated and checkpoint-specific by default: a publish should never be able
    # to land on a prefix that existing hf:// references already point into.
    prefix = args.hf_path_prefix or (
        f"bc-selected-{recommended.tag.replace('_', '-')}-"
        f"{datetime.now(timezone.utc):%Y%m%d}"
    )
    uploads = [checkpoint, stats, manifest_path]
    if run_config is not None and run_config.is_file():
        uploads.append(run_config)
    if figure_path is not None and Path(figure_path).exists():
        uploads.append(Path(figure_path))

    _assert_free(
        args.hf_repo_id,
        prefix,
        [path.name for path in uploads],
        args.hf_token,
        args.hf_allow_overwrite,
    )

    summary["hf_checkpoint_uri"] = f"hf://{args.hf_repo_id}/{prefix}/{checkpoint.name}"
    summary["hf_stats_uri"] = f"hf://{args.hf_repo_id}/{prefix}/{stats.name}"
    manifest_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")

    confirmation = summary["confirmation"]
    result = push_files_to_hub(
        repo_id=args.hf_repo_id,
        file_paths=[str(path) for path in uploads],
        private=args.hf_private,
        token=args.hf_token,
        path_prefix=prefix,
        commit_message=(
            f"BC selection: {recommended.tag} "
            f"(select mean {recommended.mean:.3f} over {recommended.n} seeds"
            + (f", confirmed {confirmation['mean']:.3f}" if confirmation else ", UNCONFIRMED")
            + ")"
        ),
    )
    print(f"\npublished {len(result.uploaded_files)} file(s) to {result.repo_url}")
    for name in result.uploaded_files:
        print(f"  {name}")
    print("\nPin this in later experiments:")
    print(f"  --bc_checkpoint {summary['hf_checkpoint_uri']} \\")
    print(f"  --bc_stats      {summary['hf_stats_uri']}")
    return result


def make_figure(cells, plateau, recommended, args) -> Path:
    import matplotlib.pyplot as plt

    apply_figure_style()
    scored = [cell for cell in cells if cell.n > 0]

    # Real numeric spacing: checkpoints are epochs, and pretending they are
    # evenly spaced categories would misstate the shape of the curve. The final
    # unsuffixed checkpoint is placed one median gap past the last numbered one.
    known = [cell.epoch for cell in scored if cell.epoch is not None]
    gaps = np.diff(sorted(known)) if len(known) > 1 else np.array([1.0])
    fallback = (max(known) + float(np.median(gaps))) if known else 0.0
    position = {
        cell.tag: (cell.epoch if cell.epoch is not None else fallback) for cell in scored
    }

    figure, axes = plt.subplots(figsize=(7.0, 4.2))

    if len(plateau) > 1:
        axes.axvspan(
            position[plateau[0].tag],
            position[plateau[-1].tag],
            color=INK["grid"],
            zorder=0,
            label=f"plateau (within {args.plateau_z} SE)",
        )

    x = [position[cell.tag] for cell in scored]

    # Per-seed points sit behind the mean and are deliberately recessive: they
    # show the spread being reasoned about without competing with the summary.
    for cell in scored:
        axes.scatter(
            [position[cell.tag]] * cell.n,
            cell.values,
            s=22,
            color=SERIES_COLORS["dream_ppo"],
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
    axes.plot(
        x,
        [cell.mean for cell in scored],
        marker="o",
        color=SERIES_COLORS["dream_ppo"],
        label=f"{SERIES_LABELS['dream_ppo']} (seed mean)",
        zorder=3,
    )

    solo = [(position[cell.tag], cell.bc_standalone) for cell in scored if cell.bc_standalone is not None]
    if solo:
        axes.plot(
            [px for px, _ in solo],
            [py for _, py in solo],
            marker="s",
            color=SERIES_COLORS["bc"],
            label=f"{SERIES_LABELS['bc']} alone",
            zorder=3,
        )

    axes.scatter(
        [position[recommended.tag]],
        [recommended.mean],
        s=140,
        facecolors="none",
        edgecolors=INK["primary"],
        linewidths=1.4,
        zorder=4,
    )
    # One direct label, on the point the reader is meant to act on.
    axes.annotate(
        f"recommended\n{recommended.tag}",
        xy=(position[recommended.tag], recommended.mean),
        # Above the point: below it runs into the BC-alone series wherever the
        # two curves cross, which is exactly the interesting part of the plot.
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=9,
        color=INK["secondary"],
    )

    axes.set_xlabel("BC training epoch")
    axes.set_ylabel(f"success rate ({recommended.episodes} held-out episodes)")
    axes.set_title("Which BC checkpoint is the best PPO initialization?", loc="left")
    axes.set_xticks(x)
    axes.set_xticklabels(
        [str(cell.epoch) if cell.epoch is not None else "final" for cell in scored]
    )
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="lower right")

    return save_figure(figure, repo_path(args.output_root) / "bc_selection.png")


if __name__ == "__main__":
    main()
