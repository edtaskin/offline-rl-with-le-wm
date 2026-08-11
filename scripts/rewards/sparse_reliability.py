"""Evaluate objective-met classifier reliability on LeWM rollouts.

This isolates whether objective-met failures come from the classifier or from
imagined latent drift. It compares:
  1. encoded GT future latents -> objective_met classifier
  2. LeWM imagined latents from GT context/actions -> objective_met classifier

The poster-oriented stratified analysis samples GT-success and GT-failure
states separately at each rollout depth, so recall/FPR are meaningful over the
whole x-axis.

No visual perturbations or invariant probes are involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rollouts import state_probes as rollout_eval  # noqa: E402


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


AXIS_COLOR = "0.35"
TITLE_COLOR = "0.22"
SPINE_COLOR = "0.65"
GRID_COLOR = "0.86"


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    if title is not None:
        ax.set_title(title, color=TITLE_COLOR, fontsize=12, fontweight="semibold", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=AXIS_COLOR, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=AXIS_COLOR, labelpad=6)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.grid(True, alpha=0.45, color=GRID_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm_1M")
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--output-dir", default="models/rollout_probe/objective_met_reliability")
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=["success_anchor", "stratified_horizon"],
        default=["success_anchor"],
    )
    parser.add_argument("--success-anchor-steps", nargs="+", type=int, default=[2, 5, 10, 16])
    parser.add_argument("--num-trajectories-per-anchor", type=int, default=512)
    parser.add_argument(
        "--stratified-samples-per-class",
        type=int,
        default=512,
        help="For stratified_horizon, sample this many GT-success and GT-failure windows at each rollout step.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Regenerate plots from existing CSV outputs without loading LeWM or probes.",
    )
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM.",
    )
    return parser.parse_args()


def load_objective_classifier(probe_dir: Path, probe_kind: str) -> rollout_eval.BinaryClassifierProbe:
    path = probe_dir / "objective_met" / f"{probe_kind}_probe.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing objective_met {probe_kind} probe: {path}")
    return rollout_eval.BinaryClassifierProbe(path)


def objective_met_mask(states: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return rollout_eval.ground_truth_targets(states, args)["objective_met"][..., 0].astype(bool)


def sample_success_anchor_starts(
    h5: h5py.File,
    anchor_step: int,
    count: int,
    args: argparse.Namespace,
) -> list[tuple[int, int]]:
    if anchor_step < 1 or anchor_step > args.horizon:
        raise ValueError(f"Anchor step {anchor_step} is outside rollout horizon {args.horizon}.")

    lengths = h5["ep_len"][:].astype(np.int64)
    offsets = h5["ep_offset"][:].astype(np.int64)
    state = h5["state"][:].astype(np.float32)
    met = objective_met_mask(state, args)
    anchor_idx = anchor_step - 1
    candidates: list[tuple[int, int]] = []

    for ep, (offset, length) in enumerate(zip(offsets, lengths)):
        offset = int(offset)
        length = int(length)
        success_locals = np.flatnonzero(met[offset : offset + length])
        for success_local in success_locals:
            start = offset + int(success_local) - (args.context_steps + anchor_idx) * args.frameskip
            first_future = start + args.context_steps * args.frameskip
            last_future = start + (args.context_steps + args.horizon - 1) * args.frameskip
            if start < offset or last_future >= offset + length:
                continue
            future_rows = first_future + np.arange(args.horizon) * args.frameskip
            first_success = np.flatnonzero(met[future_rows])
            if len(first_success) and int(first_success[0]) == anchor_idx:
                candidates.append((ep, int(start)))

    if not candidates:
        raise ValueError(f"No success-anchored windows found for anchor step {anchor_step}.")

    rng = np.random.default_rng(args.seed + 100_000 + anchor_step)
    replace = len(candidates) < count
    if replace:
        print(f"anchor {anchor_step}: {len(candidates)} candidates for {count} requested, sampling with replacement")
    chosen = rng.choice(len(candidates), size=count, replace=replace)
    return [candidates[int(i)] for i in chosen]


def sample_step_label_starts(
    offsets: np.ndarray,
    lengths: np.ndarray,
    met: np.ndarray,
    model_step: int,
    label: bool,
    count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if model_step < 1 or model_step > args.horizon:
        raise ValueError(f"model_step must be in [1, horizon], got {model_step}")

    required_last_frame = (args.context_steps + args.horizon - 1) * args.frameskip
    target_offset = (args.context_steps + model_step - 1) * args.frameskip
    ep_parts = []
    start_parts = []

    for ep, (offset, length) in enumerate(zip(offsets, lengths)):
        offset = int(offset)
        length = int(length)
        max_start_local = length - required_last_frame - 1
        if max_start_local < 0:
            continue
        local_starts = np.arange(max_start_local + 1, dtype=np.int64)
        target_rows = offset + local_starts + target_offset
        keep = met[target_rows] == label
        if not np.any(keep):
            continue
        kept_starts = offset + local_starts[keep]
        start_parts.append(kept_starts)
        ep_parts.append(np.full(len(kept_starts), ep, dtype=np.int64))

    if not start_parts:
        raise ValueError(f"No candidates found for model_step={model_step}, label={label}.")

    starts = np.concatenate(start_parts)
    eps = np.concatenate(ep_parts)
    replace = len(starts) < count
    if replace:
        print(
            f"step {model_step} label {int(label)}: "
            f"{len(starts)} candidates for {count} requested, sampling with replacement"
        )
    idx = rng.choice(len(starts), size=count, replace=replace)
    return [(int(eps[i]), int(starts[i])) for i in idx]


def classifier_curves(
    prob: np.ndarray,
    states: np.ndarray,
    args: argparse.Namespace,
    threshold: float,
) -> dict[str, np.ndarray]:
    return rollout_eval.binary_curve_metrics(prob, states, args, threshold)


def detection_delay_summary(
    prob: np.ndarray,
    states: np.ndarray,
    threshold: float,
    anchor_step: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    y = objective_met_mask(states, args)
    pred = prob[..., 0] >= threshold
    gt_first = np.full(y.shape[0], -1, dtype=np.int64)
    pred_first = np.full(y.shape[0], -1, dtype=np.int64)

    for i in range(y.shape[0]):
        gt_idx = np.flatnonzero(y[i])
        pred_idx = np.flatnonzero(pred[i])
        if len(gt_idx):
            gt_first[i] = int(gt_idx[0])
        if len(pred_idx):
            pred_first[i] = int(pred_idx[0])

    has_gt = gt_first >= 0
    detected = has_gt & (pred_first >= 0)
    early = detected & (pred_first < gt_first)
    on_time = detected & (pred_first == gt_first)
    late = detected & (pred_first > gt_first)
    missed = has_gt & (pred_first < 0)
    delay = pred_first[detected] - gt_first[detected]

    return {
        "anchor_model_step": float(anchor_step),
        "anchor_env_step": float(anchor_step * args.frameskip),
        "count": float(len(y)),
        "gt_success_count": float(has_gt.sum()),
        "detected_count": float(detected.sum()),
        "missed_count": float(missed.sum()),
        "early_count": float(early.sum()),
        "on_time_count": float(on_time.sum()),
        "late_count": float(late.sum()),
        "detection_rate": float(detected.sum() / max(has_gt.sum(), 1)),
        "miss_rate": float(missed.sum() / max(has_gt.sum(), 1)),
        "early_rate": float(early.sum() / max(has_gt.sum(), 1)),
        "on_time_rate": float(on_time.sum() / max(has_gt.sum(), 1)),
        "late_rate": float(late.sum() / max(has_gt.sum(), 1)),
        "mean_delay_wm_steps": float(np.mean(delay)) if len(delay) else float("nan"),
        "median_delay_wm_steps": float(np.median(delay)) if len(delay) else float("nan"),
    }


def write_curves_csv(path: Path, env_steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_step", "env_step", "relative_env_step", "metric", "value"])
        writer.writeheader()
        anchor_env_step = int(env_steps[np.flatnonzero(curves["gt_positive_rate"] > 0)[0]]) if np.any(curves["gt_positive_rate"] > 0) else 0
        for metric, values in sorted(curves.items()):
            for i, value in enumerate(values):
                writer.writerow(
                    {
                        "model_step": int(i + 1),
                        "env_step": int(env_steps[i]),
                        "relative_env_step": int(env_steps[i] - anchor_env_step),
                        "metric": metric,
                        "value": float(value),
                    }
                )


def write_event_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    path: Path,
    x: np.ndarray,
    curves: dict[str, np.ndarray],
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for prefix, label, style in (
        ("encoded_gt", "encoded GT future", "--"),
        ("imagined", "imagined rollout", "-"),
    ):
        key = f"{prefix}_{metric}"
        if key in curves:
            ax.plot(x, curves[key], linestyle=style, linewidth=2.4, label=label)
    ax.axvline(0, color="tab:red", linestyle=":", linewidth=2, label="first GT success")
    style_axis(
        ax,
        title=title,
        xlabel="environment steps relative to first ground-truth success",
        ylabel=ylabel,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_anchor_curves(output_dir: Path, relative_env_steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    specs = [
        ("objective_met_mean_probability", "Sparse success probability around real success", "mean success probability", (0.0, 1.0)),
        (
            "objective_met_mean_probability_on_negatives",
            "Sparse success probability before real success",
            "mean success probability on failures",
            (0.0, 1.0),
        ),
        (
            "objective_met_mean_probability_on_positives",
            "Sparse success probability after real success",
            "mean success probability on successes",
            (0.0, 1.0),
        ),
        ("objective_met_false_positive_rate", "False positives around real success", "false-positive rate", (0.0, 1.05)),
        ("objective_met_recall", "Recall around real success", "recall", (0.0, 1.05)),
        ("objective_met_precision", "Precision around real success", "precision", (0.0, 1.05)),
        ("objective_met_predicted_rate", "How often the classifier predicts success", "predicted success rate", (0.0, 1.05)),
    ]
    for metric, title, ylabel, ylim in specs:
        plot_metric(output_dir / "plots" / f"{metric}.png", relative_env_steps, curves, metric, title, ylabel, ylim)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(relative_env_steps, curves["gt_positive_rate"], color="black", linewidth=2.4)
    ax.axvline(0, color="tab:red", linestyle=":", linewidth=2)
    style_axis(
        ax,
        title="Where ground-truth success appears in the sampled rollouts",
        xlabel="environment steps relative to first ground-truth success",
        ylabel="ground-truth success rate",
    )
    ax.set_ylim(0.0, 1.05)
    path = output_dir / "plots" / "gt_positive_rate.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_stratified_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_step",
        "env_step",
        "positive_count",
        "negative_count",
        "encoded_gt_recall",
        "imagined_recall",
        "encoded_gt_fpr",
        "imagined_fpr",
        "encoded_gt_mean_p_positive",
        "imagined_mean_p_positive",
        "encoded_gt_mean_p_negative",
        "imagined_mean_p_negative",
        "latent_cosine_positive",
        "latent_cosine_negative",
        "latent_cosine_all",
        "latent_rmse_positive",
        "latent_rmse_negative",
        "latent_rmse_all",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_stratified_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def plot_existing_outputs(output_dir: Path) -> None:
    stratified_csv = output_dir / "stratified_horizon" / "stratified_recall_fpr.csv"
    if stratified_csv.exists():
        rows = read_stratified_csv(stratified_csv)
        stratified_dir = stratified_csv.parent
        plot_stratified_recall_fpr(stratified_dir / "plots" / "sparse_classifier_recall_fpr.png", rows)
        plot_stratified_recall_fpr_imagined_only(
            stratified_dir / "plots" / "sparse_classifier_recall_fpr_imagined_only.png",
            rows,
        )
        plot_stratified_support(stratified_dir / "plots" / "stratified_support.png", rows)
        plot_stratified_latent(stratified_dir / "plots" / "latent_similarity.png", rows)
        print(f"regenerated sparse stratified plots from {stratified_csv}")
    else:
        print(f"skipped stratified plots, missing {stratified_csv}")


def plot_stratified_recall_fpr(path: Path, rows: list[dict[str, float]]) -> None:
    env_steps = np.asarray([row["env_step"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(env_steps, [row["imagined_recall"] for row in rows], color="tab:blue", linewidth=2.8, label="imagined recall")
    ax.plot(
        env_steps,
        [row["encoded_gt_recall"] for row in rows],
        color="tab:blue",
        linestyle="--",
        linewidth=2.2,
        label="encoded-GT recall",
    )
    ax.plot(env_steps, [row["imagined_fpr"] for row in rows], color="tab:orange", linewidth=2.8, label="imagined FPR")
    ax.plot(
        env_steps,
        [row["encoded_gt_fpr"] for row in rows],
        color="tab:orange",
        linestyle="--",
        linewidth=2.2,
        label="encoded-GT FPR",
    )
    style_axis(
        ax,
        title="Does the sparse reward stay reliable during imagination?",
        xlabel="imagined rollout horizon (environment steps)",
        ylabel="recall / false-positive rate",
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncols=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_stratified_recall_fpr_imagined_only(path: Path, rows: list[dict[str, float]]) -> None:
    env_steps = np.asarray([row["env_step"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(env_steps, [row["imagined_recall"] for row in rows], color="tab:blue", linewidth=2.8, label="recall")
    ax.plot(env_steps, [row["imagined_fpr"] for row in rows], color="tab:orange", linewidth=2.8, label="FPR")
    style_axis(
        ax,
        title="Does the sparse reward stay reliable during imagination?",
        xlabel="imagined rollout horizon (environment steps)",
        ylabel="recall / false-positive rate",
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_stratified_support(path: Path, rows: list[dict[str, float]]) -> None:
    env_steps = np.asarray([row["env_step"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(env_steps, [row["positive_count"] for row in rows], linewidth=2, label="GT-success samples")
    ax.plot(env_steps, [row["negative_count"] for row in rows], linewidth=2, label="GT-failure samples")
    style_axis(
        ax,
        title="Samples per rollout horizon",
        xlabel="imagined rollout horizon (environment steps)",
        ylabel="number of sampled states",
    )
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stratified_latent(path: Path, rows: list[dict[str, float]]) -> None:
    env_steps = np.asarray([row["env_step"] for row in rows], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(env_steps, [row["latent_cosine_all"] for row in rows], linewidth=2.4, label="all")
    axes[0].plot(env_steps, [row["latent_cosine_positive"] for row in rows], linestyle="--", linewidth=2.0, label="GT success")
    axes[0].plot(env_steps, [row["latent_cosine_negative"] for row in rows], linestyle=":", linewidth=2.0, label="GT failure")
    style_axis(
        axes[0],
        title="How similar are imagined latents to encoded futures?",
        xlabel="imagined rollout horizon (environment steps)",
        ylabel="cosine similarity to encoded ground truth",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    axes[1].plot(env_steps, [row["latent_rmse_all"] for row in rows], linewidth=2.4, label="all")
    axes[1].plot(env_steps, [row["latent_rmse_positive"] for row in rows], linestyle="--", linewidth=2.0, label="GT success")
    axes[1].plot(env_steps, [row["latent_rmse_negative"] for row in rows], linestyle=":", linewidth=2.0, label="GT failure")
    style_axis(
        axes[1],
        title="How far do imagined latents drift from encoded futures?",
        xlabel="imagined rollout horizon (environment steps)",
        ylabel="latent RMSE to encoded ground truth",
    )
    axes[1].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)

@torch.inference_mode()
def evaluate_anchor(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    classifier: rollout_eval.BinaryClassifierProbe,
    anchor_step: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, float | str]]]:
    starts = sample_success_anchor_starts(h5, anchor_step, args.num_trajectories_per_anchor, args)
    action_mean, action_std = rollout_eval.action_stats(h5)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))
    pred_embs = []
    gt_embs = []
    state_parts = []

    for start in range(0, len(starts), args.batch_size):
        batch_starts = starts[start : start + args.batch_size]
        context_pixels, future_pixels, action_blocks, future_states = rollout_eval.load_batch(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=args.horizon,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
            normalize_actions=not args.no_normalize_actions,
        )
        pred_embs.append(
            rollout_eval.rollout_embeddings(
                model,
                context_pixels,
                action_blocks,
                horizon=args.horizon,
                history_size=history_size,
                device=device,
            )
        )
        gt_embs.append(
            rollout_eval.encode_future_embeddings(
                model,
                future_pixels,
                encode_batch_size=args.encode_batch_size,
                device=device,
            )
        )
        state_parts.append(future_states)
        print(f"anchor {anchor_step}: processed {min(start + len(batch_starts), len(starts))}/{len(starts)}")

    pred_emb = np.concatenate(pred_embs, axis=0)
    gt_emb = np.concatenate(gt_embs, axis=0)
    states = np.concatenate(state_parts, axis=0)
    imagined_prob = classifier(pred_emb)
    encoded_gt_prob = classifier(gt_emb)
    imagined_curves = classifier_curves(imagined_prob, states, args, classifier.threshold)
    encoded_gt_curves = classifier_curves(encoded_gt_prob, states, args, classifier.threshold)
    gt_positive_rate = objective_met_mask(states, args).mean(axis=0)
    curves = {
        **{f"imagined_{key}": value for key, value in imagined_curves.items()},
        **{f"encoded_gt_{key}": value for key, value in encoded_gt_curves.items()},
        "gt_positive_rate": gt_positive_rate,
        "latent_rmse": rollout_eval.latent_curve(pred_emb, gt_emb),
        "latent_cosine": rollout_eval.latent_cosine_curve(pred_emb, gt_emb),
    }
    arrays = {
        "sampled": np.asarray(starts, dtype=np.int64),
        "pred_emb": pred_emb,
        "gt_emb": gt_emb,
        "future_states": states,
        "imagined_objective_met_probability": imagined_prob,
        "encoded_gt_objective_met_probability": encoded_gt_prob,
    }
    event_rows = [
        {"source": "encoded_gt", **detection_delay_summary(encoded_gt_prob, states, classifier.threshold, anchor_step, args)},
        {"source": "imagined", **detection_delay_summary(imagined_prob, states, classifier.threshold, anchor_step, args)},
    ]
    return curves, arrays, event_rows


@torch.inference_mode()
def evaluate_starts_at_step(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    classifier: rollout_eval.BinaryClassifierProbe,
    starts: list[tuple[int, int]],
    model_step: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    history_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imagined_probs = []
    encoded_gt_probs = []
    imagined_embs = []
    encoded_gt_embs = []

    for start in range(0, len(starts), args.batch_size):
        batch_starts = starts[start : start + args.batch_size]
        context_pixels, future_pixels, action_blocks, _ = rollout_eval.load_batch(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=model_step,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
            normalize_actions=not args.no_normalize_actions,
        )
        pred_emb = rollout_eval.rollout_embeddings(
            model,
            context_pixels,
            action_blocks,
            horizon=model_step,
            history_size=history_size,
            device=device,
        )[:, -1]
        gt_emb = rollout_eval.encode_future_embeddings(
            model,
            future_pixels,
            encode_batch_size=args.encode_batch_size,
            device=device,
        )[:, -1]
        imagined_embs.append(pred_emb)
        encoded_gt_embs.append(gt_emb)
        imagined_probs.append(classifier(pred_emb, model_step=model_step)[:, 0])
        encoded_gt_probs.append(classifier(gt_emb, model_step=model_step)[:, 0])

    imagined_prob = np.concatenate(imagined_probs, axis=0)
    encoded_gt_prob = np.concatenate(encoded_gt_probs, axis=0)
    imagined_emb = np.concatenate(imagined_embs, axis=0)
    encoded_gt_emb = np.concatenate(encoded_gt_embs, axis=0)
    return imagined_prob, encoded_gt_prob, imagined_emb, encoded_gt_emb


def latent_pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    numerator = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = float(np.mean(numerator / np.maximum(denom, 1e-8)))
    return rmse, cosine


@torch.inference_mode()
def evaluate_stratified_horizon(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    classifier: rollout_eval.BinaryClassifierProbe,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    offsets = h5["ep_offset"][:].astype(np.int64)
    lengths = h5["ep_len"][:].astype(np.int64)
    state = h5["state"][:].astype(np.float32)
    met = objective_met_mask(state, args)
    action_mean, action_std = rollout_eval.action_stats(h5)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))
    rows: list[dict[str, float]] = []
    arrays: dict[str, np.ndarray] = {}

    for model_step in range(1, args.horizon + 1):
        rng = np.random.default_rng(args.seed + 200_000 + model_step)
        pos_starts = sample_step_label_starts(
            offsets,
            lengths,
            met,
            model_step,
            True,
            args.stratified_samples_per_class,
            args,
            rng,
        )
        neg_starts = sample_step_label_starts(
            offsets,
            lengths,
            met,
            model_step,
            False,
            args.stratified_samples_per_class,
            args,
            rng,
        )
        pos_imagined_p, pos_encoded_p, pos_imagined_z, pos_encoded_z = evaluate_starts_at_step(
            args, h5, model, classifier, pos_starts, model_step, action_mean, action_std, history_size, device
        )
        neg_imagined_p, neg_encoded_p, neg_imagined_z, neg_encoded_z = evaluate_starts_at_step(
            args, h5, model, classifier, neg_starts, model_step, action_mean, action_std, history_size, device
        )
        pos_rmse, pos_cos = latent_pair_metrics(pos_imagined_z, pos_encoded_z)
        neg_rmse, neg_cos = latent_pair_metrics(neg_imagined_z, neg_encoded_z)
        all_rmse, all_cos = latent_pair_metrics(
            np.concatenate([pos_imagined_z, neg_imagined_z], axis=0),
            np.concatenate([pos_encoded_z, neg_encoded_z], axis=0),
        )
        threshold = classifier.threshold
        rows.append(
            {
                "model_step": float(model_step),
                "env_step": float(model_step * args.frameskip),
                "positive_count": float(len(pos_starts)),
                "negative_count": float(len(neg_starts)),
                "encoded_gt_recall": float(np.mean(pos_encoded_p >= threshold)),
                "imagined_recall": float(np.mean(pos_imagined_p >= threshold)),
                "encoded_gt_fpr": float(np.mean(neg_encoded_p >= threshold)),
                "imagined_fpr": float(np.mean(neg_imagined_p >= threshold)),
                "encoded_gt_mean_p_positive": float(np.mean(pos_encoded_p)),
                "imagined_mean_p_positive": float(np.mean(pos_imagined_p)),
                "encoded_gt_mean_p_negative": float(np.mean(neg_encoded_p)),
                "imagined_mean_p_negative": float(np.mean(neg_imagined_p)),
                "latent_cosine_positive": pos_cos,
                "latent_cosine_negative": neg_cos,
                "latent_cosine_all": all_cos,
                "latent_rmse_positive": pos_rmse,
                "latent_rmse_negative": neg_rmse,
                "latent_rmse_all": all_rmse,
            }
        )
        arrays[f"step_{model_step}_positive_starts"] = np.asarray(pos_starts, dtype=np.int64)
        arrays[f"step_{model_step}_negative_starts"] = np.asarray(neg_starts, dtype=np.int64)
        arrays[f"step_{model_step}_positive_imagined_probability"] = pos_imagined_p
        arrays[f"step_{model_step}_positive_encoded_gt_probability"] = pos_encoded_p
        arrays[f"step_{model_step}_negative_imagined_probability"] = neg_imagined_p
        arrays[f"step_{model_step}_negative_encoded_gt_probability"] = neg_encoded_p
        print(
            f"stratified step {model_step}/{args.horizon}: "
            f"imagined recall={rows[-1]['imagined_recall']:.3f}, imagined FPR={rows[-1]['imagined_fpr']:.3f}"
        )

    return rows, arrays


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_existing:
        plot_existing_outputs(output_dir)
        return
    device = torch.device(args.device)

    probe_dir = repo_path(args.probe_dir)
    classifier = load_objective_classifier(probe_dir, args.probe_kind)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=repo_path(args.checkpoint_cache_dir))
    model = model.to(device).eval()
    model.requires_grad_(False)

    summary: dict[str, object] = {
        "config": vars(args),
        "probe_dir": str(probe_dir),
        "objective_met_threshold": classifier.threshold,
    }
    with h5py.File(repo_path(args.dataset_path), "r") as h5:
        if "success_anchor" in args.analyses:
            summary["anchors"] = {}
            for anchor_step in args.success_anchor_steps:
                anchor_step = int(anchor_step)
                curves, arrays, event_rows = evaluate_anchor(args, h5, model, classifier, anchor_step, device)
                env_steps = args.frameskip * np.arange(1, args.horizon + 1)
                relative_env_steps = env_steps - anchor_step * args.frameskip
                anchor_dir = output_dir / f"success_anchor_{anchor_step}wm"
                write_curves_csv(anchor_dir / "objective_met_curves.csv", env_steps, curves)
                write_event_summary_csv(anchor_dir / "event_summary.csv", event_rows)
                plot_anchor_curves(anchor_dir, relative_env_steps, curves)
                np.savez_compressed(
                    anchor_dir / "arrays.npz",
                    env_steps=env_steps,
                    relative_env_steps=relative_env_steps,
                    success_anchor_step=np.asarray([anchor_step], dtype=np.int64),
                    **arrays,
                    **curves,
                )
                summary["anchors"][str(anchor_step)] = {
                    "final_step": {key: float(value[-1]) for key, value in curves.items()},
                    "event_summary": event_rows,
                }
        if "stratified_horizon" in args.analyses:
            stratified_dir = output_dir / "stratified_horizon"
            rows, arrays = evaluate_stratified_horizon(args, h5, model, classifier, device)
            write_stratified_csv(stratified_dir / "stratified_recall_fpr.csv", rows)
            plot_stratified_recall_fpr(stratified_dir / "plots" / "sparse_classifier_recall_fpr.png", rows)
            plot_stratified_recall_fpr_imagined_only(
                stratified_dir / "plots" / "sparse_classifier_recall_fpr_imagined_only.png",
                rows,
            )
            plot_stratified_support(stratified_dir / "plots" / "stratified_support.png", rows)
            plot_stratified_latent(stratified_dir / "plots" / "latent_similarity.png", rows)
            np.savez_compressed(stratified_dir / "arrays.npz", **arrays)
            summary["stratified_horizon"] = {
                "csv": str(stratified_dir / "stratified_recall_fpr.csv"),
                "poster_plot": str(stratified_dir / "plots" / "sparse_classifier_recall_fpr.png"),
                "rows": rows,
            }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved objective-met rollout reliability analysis to {output_dir}")


if __name__ == "__main__":
    main()
