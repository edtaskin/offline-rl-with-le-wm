"""RQ2, part 2: how far into imagination can the success probe be trusted?

The dream reward and the dream episode's *termination* both come from one frozen
binary probe reading a latent the simulator never corrected. If that probe's
false-positive rate grows with imagination depth, PPO is being paid to walk into
the part of the horizon where the probe is wrong -- the model-exploitation story.

This script measures exactly that, then draws it against the horizon the trainer
actually uses:

1. It resolves the ``objective_met`` classifier the *same way*
   :class:`src.ppo.train_lewm.LeWMDreamWorld` does, and pins
   ``scripts/rollouts/state_probes.py`` to that exact file. A curve about a
   different probe than the one PPO optimized against would prove nothing.
2. It rolls LeWM forward from ground-truth context with ground-truth expert
   actions -- so the reference state is known at every step -- and reads the
   probe at each imagined step, out to ``--horizon`` predictor steps
   (default 20 x frameskip 5 = 5..100 env steps).
3. It plots the imagined false-positive rate against the *encoded ground-truth*
   false-positive rate. The gap between them separates the two error sources:
   the encoded-GT curve is the probe being wrong on a correct latent, and
   anything above it is the world model's own drift.
4. It overlays the dream episode horizon and the mean imagined episode length
   observed during training, so the curve is read where the agent actually lives.

Usage::

    python -m scripts.rq2.probe_horizon --seeds 1 2 3
    python -m scripts.rq2.probe_horizon --skip-rollout   # re-plot existing CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
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
    apply_figure_style,
    find_run_dir,
    read_selection_log,
    repo_path,
    save_figure,
)

# Mirrors _StateProbe.find in src/ppo/train_lewm.py: first match wins.
SUCCESS_PROBE_CANDIDATES = (
    "objective_met/mlp_probe.pt",
    "is_objective_met_probe_baseline.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm")
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="imagination depth in predictor steps (x frameskip = env steps; default 20 -> 100)",
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-trajectories", type=int, default=512)
    parser.add_argument("--rollout-seed", type=int, default=3072)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--dream-exp", default="latent_ppo_pusht_lewm_sparse_dense_correlation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="runs/rq2")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="reuse the existing rollout_probe_curves.csv instead of re-running the sweep",
    )
    return parser.parse_args()


def resolve_success_probe(probe_dir: Path) -> Path:
    for relative in SUCCESS_PROBE_CANDIDATES:
        path = probe_dir / relative
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No objective_met classifier under {probe_dir}; looked for "
        + ", ".join(SUCCESS_PROBE_CANDIDATES)
        + ". This is the same probe src/ppo/train_lewm.py loads, so if it is "
        "missing the dream trainer cannot have run either."
    )


def run_probe_rollouts(args, classifier: Path, output_dir: Path) -> None:
    argv = [
        sys.executable,
        "-m",
        "scripts.rollouts.state_probes",
        "--probe-dir", str(args.probe_dir),
        "--classifier-checkpoint", str(classifier),
        "--dataset-path", args.dataset_path,
        "--horizon", str(args.horizon),
        "--frameskip", str(args.frameskip),
        "--num-trajectories", str(args.num_trajectories),
        "--seed", str(args.rollout_seed),
        "--output-dir", str(output_dir),
    ]
    if args.device:
        argv += ["--device", args.device]
    print("$ " + " ".join(argv))
    subprocess.run(argv, check=True, cwd=REPO_ROOT)


def read_curves(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def _cap_from_checkpoint(run_dir: Path, default_frameskip: int) -> float | None:
    """Horizon cap read from a run's own config, for logs written before it was logged."""
    import torch

    for name in ("final.pt", "latest.pt", "best.pt"):
        path = run_dir / name
        if not path.exists():
            continue
        config = torch.load(path, map_location="cpu", weights_only=False).get("config", {})
        steps = config.get("dream_eval_steps") or config.get("dream_episode_steps")
        if steps:
            return float(steps) * float(config.get("wm_frameskip", default_frameskip))
    return None


def dream_episode_facts(args) -> dict:
    """Horizon cap and observed imagined episode length, from the training runs."""
    caps, observed = [], []
    for seed in args.seeds:
        try:
            run = find_run_dir(args.dream_exp, seed, args.runs_root)
            rows = read_selection_log(run.path)
        except FileNotFoundError:
            continue
        for row in rows:
            frameskip = row.get("wm_frameskip") or args.frameskip
            if row.get("dream_episode_steps"):
                caps.append(row["dream_episode_steps"] * frameskip)
            if row.get("dream_length_env_steps"):
                observed.append(row["dream_length_env_steps"])
        if not caps:
            # Selection logs from before the horizon fields were recorded still
            # carry the answer in the checkpoint config.
            cap = _cap_from_checkpoint(run.path, args.frameskip)
            if cap:
                caps.append(cap)
    return {
        "horizon_cap_env_steps": float(np.median(caps)) if caps else None,
        "mean_observed_length_env_steps": float(np.mean(observed)) if observed else None,
        "observed_length_range": (
            [float(np.min(observed)), float(np.max(observed))] if observed else None
        ),
    }


def plot_horizon(curves, facts, output_root: Path, args) -> dict:
    env_steps = curves["env_step"]
    imagined = curves["imagined_objective_met_false_positive_rate"]
    encoded_gt = curves["encoded_gt_objective_met_false_positive_rate"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(env_steps, imagined, color=SERIES_COLORS["dream"], linewidth=2.2)
    ax.plot(
        env_steps,
        encoded_gt,
        color=SERIES_COLORS["real"],
        linewidth=2.0,
        linestyle=(0, (5, 3)),
    )
    ax.fill_between(
        env_steps, encoded_gt, imagined, color=SERIES_COLORS["dream"], alpha=0.10, linewidth=0
    )
    ax.annotate(
        "imagined latent",
        xy=(env_steps[-1], imagined[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=SERIES_COLORS["dream"],
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        "encoded ground-truth latent\n(probe error alone)",
        xy=(env_steps[-1], encoded_gt[-1]),
        xytext=(6, -14),
        textcoords="offset points",
        color=SERIES_COLORS["real"],
        fontsize=9,
    )

    cap = facts.get("horizon_cap_env_steps")
    if cap:
        ax.axvline(cap, color=INK["muted"], linewidth=1.2, linestyle=(0, (2, 3)))
        ax.annotate(
            f"dream episode cap\n{cap:.0f} env steps",
            xy=(cap, ax.get_ylim()[1]),
            xytext=(-6, -26),
            textcoords="offset points",
            ha="right",
            color=INK["secondary"],
            fontsize=8,
        )
    observed = facts.get("mean_observed_length_env_steps")
    if observed:
        ax.axvspan(0, observed, color=INK["grid"], alpha=0.55, zorder=0)
        ax.annotate(
            f"where PPO actually lives\n(mean imagined episode: {observed:.0f} env steps)",
            xy=(observed, 0),
            xytext=(6, 12),
            textcoords="offset points",
            color=INK["secondary"],
            fontsize=8,
        )

    ax.set_xlabel("imagination horizon (environment steps after context)")
    ax.set_ylabel("objective-met false-positive rate")
    ax.set_xlim(0, env_steps[-1] * 1.02)
    ax.set_ylim(bottom=0)
    ax.set_title("Probe-declared success gets less trustworthy with depth", pad=12)
    fig.text(
        0.5,
        -0.04,
        "shaded band = error the world model adds on top of the probe's own error",
        ha="center",
        color=INK["muted"],
        fontsize=8,
    )
    save_figure(fig, output_root / "rq2_probe_horizon_fpr.png")
    plt.close(fig)

    def at(env_step):
        index = int(np.argmin(np.abs(env_steps - env_step)))
        return {
            "env_step": float(env_steps[index]),
            "imagined_fpr": float(imagined[index]),
            "encoded_gt_fpr": float(encoded_gt[index]),
            "imagined_precision": float(
                curves["imagined_objective_met_precision"][index]
            ),
            "imagined_recall": float(curves["imagined_objective_met_recall"][index]),
            "imagined_predicted_rate": float(
                curves["imagined_objective_met_predicted_rate"][index]
            ),
            "true_positive_rate_of_declared": float(
                curves["imagined_objective_met_precision"][index]
            ),
        }

    marks = [5, 25, 50, 75, 100]
    if observed:
        marks.append(observed)
    if cap:
        marks.append(cap)
    return {
        "at_horizon": {f"{mark:.0f}_env_steps": at(mark) for mark in sorted(set(marks))},
        "final": at(env_steps[-1]),
    }


def main() -> None:
    args = parse_args()
    apply_figure_style()
    output_root = repo_path(args.output_root)
    rollout_dir = output_root / "probe_rollouts"
    curves_path = rollout_dir / "rollout_probe_curves.csv"

    classifier = resolve_success_probe(repo_path(args.probe_dir))
    print(f"objective_met classifier (as loaded by the dream trainer): {classifier}")

    if not args.skip_rollout:
        run_probe_rollouts(args, classifier, rollout_dir)
    if not curves_path.exists():
        raise SystemExit(f"{curves_path} not found; drop --skip-rollout to generate it")

    curves = read_curves(curves_path)
    facts = dream_episode_facts(args)
    if facts["horizon_cap_env_steps"] is None:
        print(
            "note: no dream training run found, so the horizon overlay is omitted. "
            "The false-positive curve itself is unaffected."
        )
    summary = plot_horizon(curves, facts, output_root / "figures", args)
    summary["dream_episode"] = facts
    summary["classifier_checkpoint"] = str(classifier)
    summary["rollout"] = {
        "horizon_predictor_steps": args.horizon,
        "frameskip": args.frameskip,
        "num_trajectories": args.num_trajectories,
        "seed": args.rollout_seed,
    }

    path = output_root / "probe_horizon.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")

    first, last = summary["at_horizon"]["5_env_steps"], summary["final"]
    print(
        f"\nFalse-positive rate of probe-declared success:\n"
        f"  at   5 env steps of imagination: {first['imagined_fpr']:.4f} "
        f"(probe alone: {first['encoded_gt_fpr']:.4f})\n"
        f"  at {last['env_step']:3.0f} env steps of imagination: {last['imagined_fpr']:.4f} "
        f"(probe alone: {last['encoded_gt_fpr']:.4f})"
    )


if __name__ == "__main__":
    main()
