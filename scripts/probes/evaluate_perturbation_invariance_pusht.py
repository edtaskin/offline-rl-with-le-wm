"""Evaluate PushT LeWM probe robustness to visual perturbations.

Two analyses are produced:
  1. direct: clean frame vs perturbed same frame, encoded independently.
  2. rollout: clean context vs perturbed context, rolled forward with the same
     GT action blocks and compared over imagined rollout time.

Relative paths are resolved from the top-level wrapper repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes import probe_rollouts_pusht as probe_eval  # noqa: E402
from scripts.probes.train_invariant_probes_pusht import (  # noqa: E402
    DEFAULT_PERTURBATIONS,
    PERTURBATION_CHOICES,
    apply_perturbation,
)


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
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm_1M_invariant")
    parser.add_argument("--probe-kind", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--num-trajectories", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=["direct", "rollout", "encoded_rollout"],
        default=["direct", "rollout"],
        help=(
            "direct encodes clean/perturbed single frames; rollout generates clean/perturbed imagined rollouts; "
            "encoded_rollout appends an encoded-GT-future baseline to an existing rollout/arrays.npz."
        ),
    )
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Regenerate plots from existing direct/metrics.csv and rollout/metrics.csv without loading LeWM or probes.",
    )
    parser.add_argument(
        "--clean-plots",
        action="store_true",
        help="Delete existing direct/plots and rollout/plots before regenerating plots.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=PERTURBATION_CHOICES,
        default=DEFAULT_PERTURBATIONS,
    )
    parser.add_argument("--agent-color", nargs=3, type=int, default=[220, 40, 40])
    parser.add_argument("--block-color", nargs=3, type=int, default=[150, 90, 220])
    parser.add_argument("--target-color", nargs=3, type=int, default=[255, 185, 60])
    parser.add_argument("--brightness-delta", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=12.0)
    parser.add_argument("--blur-radius", type=float, default=1.25)
    parser.add_argument("--preview-row-index", type=int, default=None)
    parser.add_argument(
        "--preview-random",
        action="store_true",
        help="Use a random frame for perturbation example images instead of the first sampled frame.",
    )
    parser.add_argument("--color-min", type=float, default=0.25)
    parser.add_argument("--color-margin", type=float, default=0.05)
    parser.add_argument("--gray-min", type=float, default=0.20)
    parser.add_argument("--gray-max", type=float, default=0.85)
    parser.add_argument("--gray-chroma", type=float, default=0.18)
    parser.add_argument("--mask-border-margin", type=int, default=8)
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM rollouts.",
    )
    return parser.parse_args()


def sample_rows(h5: h5py.File, count: int, seed: int) -> np.ndarray:
    total = int(h5["pixels"].shape[0])
    rng = np.random.default_rng(seed)
    if count < 0 or count >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    breaks = np.nonzero(np.diff(rows) != 1)[0] + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(rows, breaks)]


def iter_direct_batches(h5: h5py.File, rows: np.ndarray, batch_size: int):
    pixel_parts = []
    state_parts = []
    row_parts = []
    n_buffered = 0
    for run_start, run_end in contiguous_runs(rows):
        for start in range(run_start, run_end, batch_size):
            end = min(start + batch_size, run_end)
            pixel_parts.append(h5["pixels"][start:end])
            state_parts.append(h5["state"][start:end].astype(np.float32))
            row_parts.append(np.arange(start, end, dtype=np.int64))
            n_buffered += end - start
            if n_buffered >= batch_size:
                pixels = np.concatenate(pixel_parts, axis=0)
                states = np.concatenate(state_parts, axis=0)
                batch_rows = np.concatenate(row_parts, axis=0)
                yield pixels[:batch_size], states[:batch_size], batch_rows[:batch_size]
                pixel_parts = [pixels[batch_size:]] if len(pixels[batch_size:]) else []
                state_parts = [states[batch_size:]] if len(states[batch_size:]) else []
                row_parts = [batch_rows[batch_size:]] if len(batch_rows[batch_size:]) else []
                n_buffered = len(pixel_parts[0]) if pixel_parts else 0
    if n_buffered:
        yield np.concatenate(pixel_parts, axis=0), np.concatenate(state_parts, axis=0), np.concatenate(row_parts, axis=0)


def perturb_frame_batch(pixels: np.ndarray, perturbation: str, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    return np.stack([apply_perturbation(frame, perturbation, args, rng) for frame in pixels], axis=0)


def perturb_context(context: np.ndarray, perturbation: str, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    flat = context.reshape(-1, *context.shape[2:])
    pert = perturb_frame_batch(flat, perturbation, args, rng)
    return pert.reshape(context.shape)


def display_name(name: str) -> str:
    if name == "clean":
        return "Clean"
    return name.replace("_", " ").title()


def choose_preview_frame(h5: h5py.File, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    if args.preview_row_index is not None:
        row = int(args.preview_row_index)
    elif args.preview_random:
        row = int(np.random.default_rng(args.seed + 9000).integers(0, int(h5["pixels"].shape[0])))
    else:
        row = int(sample_rows(h5, 1, args.seed)[0])
    return h5["pixels"][row], row


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_label(
    draw: ImageDraw.ImageDraw,
    x0: int,
    width: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    tw, _ = text_size(draw, text, font)
    draw.text((x0 + (width - tw) / 2, y), text, font=font, fill=fill)


def plot_perturbation_examples(output_dir: Path, h5: h5py.File, args: argparse.Namespace) -> None:
    frame, row = choose_preview_frame(h5, args)
    names = ["clean", *args.perturbations]
    rng = np.random.default_rng(args.seed + 9100 + row)
    images = [frame] + [apply_perturbation(frame, perturbation, args, rng) for perturbation in args.perturbations]

    example_dir = output_dir / "perturbation_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    cols = len(names)
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.5), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    for ax, name, image in zip(axes, names, images):
        ax.imshow(image)
        ax.set_title(display_name(name), fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Perturbation Examples: dataset row {row}", fontsize=11)
    fig.savefig(example_dir / "perturbation_grid.png", dpi=180)
    plt.close(fig)

    pil_images = [Image.fromarray(image) for image in images]
    w, h = pil_images[0].size
    top_h = 30
    bottom_h = 30
    canvas = Image.new("RGB", (w * cols, top_h + h + bottom_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (name, image) in enumerate(zip(names, pil_images)):
        x = i * w
        label = display_name(name)
        draw_centered_label(draw, x, w, 8, label, font)
        canvas.paste(image, (x, top_h))
        draw_centered_label(draw, x, w, top_h + h + 8, label, font)
    canvas.save(example_dir / "perturbation_horizontal_strip.png")
    with (example_dir / "metadata.json").open("w") as f:
        json.dump({"row": row, "perturbations": names}, f, indent=2)


@torch.inference_mode()
def encode_frames(model: torch.nn.Module, pixels: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks = []
    for start in range(0, len(pixels), batch_size):
        chunk = pixels[start : start + batch_size]
        x = probe_eval.preprocess_pixels(chunk[:, None], device)
        emb = model.encode({"pixels": x})["emb"][:, 0]
        chunks.append(emb.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def latent_rmse_flat(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def latent_cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    numerator = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.mean(numerator / np.maximum(denom, 1e-8)))


def direct_feature_metrics(
    clean_pred: dict[str, np.ndarray],
    pert_pred: dict[str, np.ndarray],
    clean_cls: dict[str, np.ndarray],
    pert_cls: dict[str, np.ndarray],
    states: np.ndarray,
    args: argparse.Namespace,
    threshold: float,
) -> dict[str, float]:
    gt = probe_eval.ground_truth_targets(states, args)
    metrics: dict[str, float] = {}
    for feature in probe_eval.REGRESSION_FEATURES:
        c = clean_pred[feature]
        p = pert_pred[feature]
        y = gt[feature]
        if feature == "block_angle":
            clean_err = probe_eval.sincos_diff_rad(c, y)
            pert_err = probe_eval.sincos_diff_rad(p, y)
            diff = probe_eval.sincos_diff_rad(p, c)
            metrics[f"clean_{feature}_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(clean_err**2))))
            metrics[f"perturbed_{feature}_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(pert_err**2))))
            metrics[f"excess_{feature}_rmse_deg"] = metrics[f"perturbed_{feature}_rmse_deg"] - metrics[f"clean_{feature}_rmse_deg"]
            metrics[f"prediction_delta_{feature}_deg"] = float(np.degrees(np.mean(np.abs(diff))))
        elif feature in ("block_rel_objective", "block_rel_agent"):
            clean_xy = c[..., :2] - y[..., :2]
            pert_xy = p[..., :2] - y[..., :2]
            delta_xy = p[..., :2] - c[..., :2]
            clean_ang = probe_eval.sincos_diff_rad(c[..., 2:4], y[..., 2:4])
            pert_ang = probe_eval.sincos_diff_rad(p[..., 2:4], y[..., 2:4])
            delta_ang = probe_eval.sincos_diff_rad(p[..., 2:4], c[..., 2:4])
            metrics[f"clean_{feature}_xy_rmse_px"] = float(np.sqrt(np.mean(clean_xy**2)))
            metrics[f"perturbed_{feature}_xy_rmse_px"] = float(np.sqrt(np.mean(pert_xy**2)))
            metrics[f"excess_{feature}_xy_rmse_px"] = metrics[f"perturbed_{feature}_xy_rmse_px"] - metrics[f"clean_{feature}_xy_rmse_px"]
            metrics[f"prediction_delta_{feature}_xy_px"] = float(np.sqrt(np.mean(delta_xy**2)))
            metrics[f"clean_{feature}_angle_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(clean_ang**2))))
            metrics[f"perturbed_{feature}_angle_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(pert_ang**2))))
            metrics[f"excess_{feature}_angle_rmse_deg"] = metrics[f"perturbed_{feature}_angle_rmse_deg"] - metrics[f"clean_{feature}_angle_rmse_deg"]
            metrics[f"prediction_delta_{feature}_angle_deg"] = float(np.degrees(np.mean(np.abs(delta_ang))))
        else:
            clean_err = c - y
            pert_err = p - y
            delta = p - c
            metrics[f"clean_{feature}_rmse_px"] = float(np.sqrt(np.mean(clean_err**2)))
            metrics[f"perturbed_{feature}_rmse_px"] = float(np.sqrt(np.mean(pert_err**2)))
            metrics[f"excess_{feature}_rmse_px"] = metrics[f"perturbed_{feature}_rmse_px"] - metrics[f"clean_{feature}_rmse_px"]
            metrics[f"prediction_delta_{feature}_px"] = float(np.sqrt(np.mean(delta**2)))

    for prefix, prob in (("clean", clean_cls["objective_met"]), ("perturbed", pert_cls["objective_met"])):
        curves = probe_eval.binary_curve_metrics(prob[:, None, :], states[:, None, :], args, threshold)
        for key, value in curves.items():
            metrics[f"{prefix}_{key}"] = float(value[0])
    metrics["excess_objective_met_false_positive_rate"] = (
        metrics["perturbed_objective_met_false_positive_rate"] - metrics["clean_objective_met_false_positive_rate"]
    )
    metrics["excess_objective_met_recall"] = metrics["perturbed_objective_met_recall"] - metrics["clean_objective_met_recall"]
    metrics["prediction_delta_objective_met_probability"] = float(
        np.mean(np.abs(pert_cls["objective_met"] - clean_cls["objective_met"]))
    )
    return metrics


def evaluate_direct(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    rows = sample_rows(h5, args.num_samples, args.seed + 11)
    threshold = classifier_probes["objective_met"].threshold
    results = {}
    arrays = {"rows": rows}

    for perturbation in args.perturbations:
        rng = np.random.default_rng(args.seed + 100_000 + args.perturbations.index(perturbation))
        clean_z_parts = []
        pert_z_parts = []
        state_parts = []
        for pixels, states, _ in iter_direct_batches(h5, rows, args.batch_size):
            pert_pixels = perturb_frame_batch(pixels, perturbation, args, rng)
            clean_z_parts.append(encode_frames(model, pixels, args.encode_batch_size, device))
            pert_z_parts.append(encode_frames(model, pert_pixels, args.encode_batch_size, device))
            state_parts.append(states)
        clean_z = np.concatenate(clean_z_parts, axis=0)
        pert_z = np.concatenate(pert_z_parts, axis=0)
        states = np.concatenate(state_parts, axis=0)
        clean_pred = probe_eval.probe_predictions(probes, clean_z)
        pert_pred = probe_eval.probe_predictions(probes, pert_z)
        clean_cls = probe_eval.classifier_predictions(classifier_probes, clean_z)
        pert_cls = probe_eval.classifier_predictions(classifier_probes, pert_z)
        metrics = direct_feature_metrics(clean_pred, pert_pred, clean_cls, pert_cls, states, args, threshold)
        metrics["latent_rmse"] = latent_rmse_flat(pert_z, clean_z)
        metrics["latent_cosine"] = latent_cosine_flat(pert_z, clean_z)
        results[perturbation] = metrics
        arrays[f"{perturbation}_clean_z"] = clean_z
        arrays[f"{perturbation}_perturbed_z"] = pert_z
        print(f"direct {perturbation}: latent_cosine={metrics['latent_cosine']:.4f}")
    arrays["states"] = states
    return results, arrays


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:]
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def rollout_curve_metrics(
    clean_z: np.ndarray,
    pert_z: np.ndarray,
    states: np.ndarray,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    clean_pred = probe_eval.probe_predictions(probes, clean_z)
    pert_pred = probe_eval.probe_predictions(probes, pert_z)
    clean_cls = probe_eval.classifier_predictions(classifier_probes, clean_z)
    pert_cls = probe_eval.classifier_predictions(classifier_probes, pert_z)
    clean_curves = probe_eval.compute_curves(clean_pred, states, args)
    pert_curves = probe_eval.compute_curves(pert_pred, states, args)
    threshold = classifier_probes["objective_met"].threshold
    clean_curves.update(probe_eval.binary_curve_metrics(clean_cls["objective_met"], states, args, threshold))
    pert_curves.update(probe_eval.binary_curve_metrics(pert_cls["objective_met"], states, args, threshold))
    excess = probe_eval.excess_curves(pert_curves, clean_curves)
    curves = {
        **{f"clean_{key}": value for key, value in clean_curves.items()},
        **{f"perturbed_{key}": value for key, value in pert_curves.items()},
        **{f"excess_{key}": value for key, value in excess.items()},
        "latent_rmse": probe_eval.latent_curve(pert_z, clean_z),
        "latent_cosine": probe_eval.latent_cosine_curve(pert_z, clean_z),
    }
    return curves


def encoded_gt_curve_metrics(
    encoded_z: np.ndarray,
    states: np.ndarray,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    pred = probe_eval.probe_predictions(probes, encoded_z)
    cls = probe_eval.classifier_predictions(classifier_probes, encoded_z)
    curves = probe_eval.compute_curves(pred, states, args)
    threshold = classifier_probes["objective_met"].threshold
    curves.update(probe_eval.binary_curve_metrics(cls["objective_met"], states, args, threshold))
    return curves


def append_encoded_rollout_baseline(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    device: torch.device,
    rollout_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    arrays_path = rollout_dir / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(
            f"Missing existing rollout arrays at {arrays_path}. Run --analyses rollout first, then rerun encoded_rollout."
        )

    existing = np.load(arrays_path, allow_pickle=False)
    arrays = {key: existing[key] for key in existing.files}
    starts = [tuple(map(int, pair)) for pair in arrays["sampled"]]
    env_steps = arrays.get("env_steps", args.frameskip * np.arange(1, args.horizon + 1))
    horizon = int(len(env_steps))
    future_states = arrays["future_states"].astype(np.float32)

    encoded_parts = []
    action_mean, action_std = action_stats(h5)
    for start in range(0, len(starts), args.batch_size):
        batch_starts = starts[start : start + args.batch_size]
        _, future_pixels, _, _ = probe_eval.load_batch(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=horizon,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
            normalize_actions=not args.no_normalize_actions,
        )
        encoded_parts.append(probe_eval.encode_future_embeddings(model, future_pixels, args.encode_batch_size, device))
        print(f"encoded GT rollout baseline: processed {min(start + len(batch_starts), len(starts))}/{len(starts)}")

    encoded_z = np.concatenate(encoded_parts, axis=0)
    encoded_curves = encoded_gt_curve_metrics(encoded_z, future_states, probes, classifier_probes, args)
    arrays["encoded_gt_z"] = encoded_z

    results: dict[str, dict[str, np.ndarray]] = {}
    for perturbation in args.perturbations:
        clean_key = f"{perturbation}_clean_z"
        pert_key = f"{perturbation}_perturbed_z"
        if clean_key not in arrays or pert_key not in arrays:
            print(f"skipping {perturbation}: missing cached clean/perturbed rollout latents")
            continue

        clean_z = arrays[clean_key]
        pert_z = arrays[pert_key]
        clean_pred = probe_eval.probe_predictions(probes, clean_z)
        pert_pred = probe_eval.probe_predictions(probes, pert_z)
        clean_cls = probe_eval.classifier_predictions(classifier_probes, clean_z)
        pert_cls = probe_eval.classifier_predictions(classifier_probes, pert_z)
        clean_curves = probe_eval.compute_curves(clean_pred, future_states, args)
        pert_curves = probe_eval.compute_curves(pert_pred, future_states, args)
        threshold = classifier_probes["objective_met"].threshold
        clean_curves.update(probe_eval.binary_curve_metrics(clean_cls["objective_met"], future_states, args, threshold))
        pert_curves.update(probe_eval.binary_curve_metrics(pert_cls["objective_met"], future_states, args, threshold))

        clean_minus_encoded = probe_eval.excess_curves(clean_curves, encoded_curves)
        pert_minus_encoded = probe_eval.excess_curves(pert_curves, encoded_curves)
        pert_minus_clean = probe_eval.excess_curves(pert_curves, clean_curves)
        curves = {
            **{f"encoded_gt_{key}": value for key, value in encoded_curves.items()},
            **{f"clean_{key}": value for key, value in clean_curves.items()},
            **{f"perturbed_{key}": value for key, value in pert_curves.items()},
            **{f"excess_{key}": value for key, value in pert_minus_clean.items()},
            **{f"clean_minus_encoded_gt_{key}": value for key, value in clean_minus_encoded.items()},
            **{f"perturbed_minus_encoded_gt_{key}": value for key, value in pert_minus_encoded.items()},
            "clean_latent_rmse_to_encoded_gt": probe_eval.latent_curve(clean_z, encoded_z),
            "clean_latent_cosine_to_encoded_gt": probe_eval.latent_cosine_curve(clean_z, encoded_z),
            "perturbed_latent_rmse_to_encoded_gt": probe_eval.latent_curve(pert_z, encoded_z),
            "perturbed_latent_cosine_to_encoded_gt": probe_eval.latent_cosine_curve(pert_z, encoded_z),
            "latent_rmse": probe_eval.latent_curve(pert_z, clean_z),
            "latent_cosine": probe_eval.latent_cosine_curve(pert_z, clean_z),
        }
        results[perturbation] = curves
        for metric, values in curves.items():
            arrays[f"{perturbation}_{metric}"] = values

    arrays["env_steps"] = env_steps
    return results, arrays


def evaluate_rollout(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    action_mean, action_std = action_stats(h5)
    starts = probe_eval.sample_starts(h5, args.num_trajectories, args.context_steps, args.horizon, args.frameskip, args.seed)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))
    results = {}
    arrays = {"sampled": np.asarray(starts, dtype=np.int64)}

    for perturbation in args.perturbations:
        rng = np.random.default_rng(args.seed + 200_000 + args.perturbations.index(perturbation))
        clean_parts = []
        pert_parts = []
        state_parts = []
        for start in range(0, len(starts), args.batch_size):
            batch_starts = starts[start : start + args.batch_size]
            context_pixels, _, action_blocks, future_states = probe_eval.load_batch(
                h5,
                batch_starts,
                context_steps=args.context_steps,
                horizon=args.horizon,
                frameskip=args.frameskip,
                action_mean=action_mean,
                action_std=action_std,
                normalize_actions=not args.no_normalize_actions,
            )
            pert_context = perturb_context(context_pixels, perturbation, args, rng)
            clean_parts.append(
                probe_eval.rollout_embeddings(model, context_pixels, action_blocks, args.horizon, history_size, device)
            )
            pert_parts.append(
                probe_eval.rollout_embeddings(model, pert_context, action_blocks, args.horizon, history_size, device)
            )
            state_parts.append(future_states)
            print(f"rollout {perturbation}: processed {min(start + len(batch_starts), len(starts))}/{len(starts)}")
        clean_z = np.concatenate(clean_parts, axis=0)
        pert_z = np.concatenate(pert_parts, axis=0)
        states = np.concatenate(state_parts, axis=0)
        curves = rollout_curve_metrics(clean_z, pert_z, states, probes, classifier_probes, args)
        results[perturbation] = curves
        arrays[f"{perturbation}_clean_z"] = clean_z
        arrays[f"{perturbation}_perturbed_z"] = pert_z
    arrays["future_states"] = states
    return results, arrays


def write_direct_csv(path: Path, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = sorted({metric for values in results.values() for metric in values})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", *metrics])
        writer.writeheader()
        for perturbation, values in results.items():
            writer.writerow({"perturbation": perturbation, **values})


def read_direct_csv(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perturbation = str(row["perturbation"])
            results[perturbation] = {
                key: float(value)
                for key, value in row.items()
                if key != "perturbation" and value not in ("", None)
            }
    return results


def write_rollout_csv(path: Path, env_steps: np.ndarray, results: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", "model_step", "env_step", "metric", "value"])
        writer.writeheader()
        for perturbation, curves in results.items():
            for metric, values in sorted(curves.items()):
                for i, value in enumerate(values):
                    writer.writerow(
                        {
                            "perturbation": perturbation,
                            "model_step": int(i + 1),
                            "env_step": int(env_steps[i]),
                            "metric": metric,
                            "value": float(value),
                        }
                    )


def read_rollout_csv(path: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    values: dict[str, dict[str, dict[int, float]]] = {}
    env_by_step: dict[int, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perturbation = str(row["perturbation"])
            metric = str(row["metric"])
            model_step = int(row["model_step"])
            env_by_step[model_step] = int(float(row["env_step"]))
            values.setdefault(perturbation, {}).setdefault(metric, {})[model_step] = float(row["value"])

    steps = sorted(env_by_step)
    env_steps = np.asarray([env_by_step[step] for step in steps], dtype=np.int64)
    results: dict[str, dict[str, np.ndarray]] = {}
    for perturbation, metric_values in values.items():
        results[perturbation] = {
            metric: np.asarray([by_step.get(step, np.nan) for step in steps], dtype=np.float32)
            for metric, by_step in metric_values.items()
        }
    return env_steps, results


def write_event_summary_csv(path: Path, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = sorted({metric for values in results.values() for metric in values})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", *metrics])
        writer.writeheader()
        for perturbation, values in results.items():
            writer.writerow({"perturbation": perturbation, **values})


def clean_plot_dirs(output_dir: Path) -> None:
    for path in (
        output_dir / "direct" / "plots",
        output_dir / "rollout" / "plots",
    ):
        if path.exists():
            shutil.rmtree(path)


def plot_existing_outputs(output_dir: Path, clean_plots: bool) -> None:
    if clean_plots:
        clean_plot_dirs(output_dir)

    direct_metrics = output_dir / "direct" / "metrics.csv"
    if direct_metrics.exists():
        plot_direct_bars(output_dir / "direct", read_direct_csv(direct_metrics))
        print(f"regenerated direct plots from {direct_metrics}")
    else:
        print(f"skipped direct plots, missing {direct_metrics}")

    rollout_metrics = output_dir / "rollout" / "metrics.csv"
    if rollout_metrics.exists():
        env_steps, rollout_results = read_rollout_csv(rollout_metrics)
        plot_rollout_curves(output_dir / "rollout", env_steps, rollout_results)
        plot_rollout_baseline_comparisons(output_dir / "rollout", env_steps, rollout_results)
        print(f"regenerated rollout plots from {rollout_metrics}")
    else:
        print(f"skipped rollout plots, missing {rollout_metrics}")


def plot_direct_bars(output_dir: Path, results: dict[str, dict[str, float]]) -> None:
    labels = list(results)

    latent_specs = [
        ("latent_cosine", "How much do perturbations alter latent geometry?", "cosine similarity to clean encoding"),
        ("latent_rmse", "How far do perturbations move encoded latents?", "latent RMSE to clean encoding"),
    ]
    for metric, title, ylabel in latent_specs:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        values = [results[label].get(metric, np.nan) for label in labels]
        ax.bar(labels, values)
        style_axis(ax, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=35)
        if metric == "latent_cosine":
            ax.set_ylim(0.0, 1.0)
        path = output_dir / "plots" / "latent" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    excess_specs = [
        ("excess_agent_pos_rmse_px", "How much do perturbations hurt agent-state recovery?", "extra error over clean encoding (px)"),
        ("excess_block_pos_rmse_px", "How much do perturbations hurt block-state recovery?", "extra error over clean encoding (px)"),
        ("excess_block_angle_rmse_deg", "How much do perturbations hurt block orientation?", "extra error over clean encoding (degrees)"),
        ("excess_block_rel_objective_xy_rmse_px", "How much do perturbations hurt T-to-goal geometry?", "extra error over clean encoding (px)"),
        (
            "excess_block_rel_objective_angle_rmse_deg",
            "How much do perturbations hurt T-to-goal orientation?",
            "extra error over clean encoding (degrees)",
        ),
        ("excess_block_rel_agent_xy_rmse_px", "How much do perturbations hurt T-to-agent geometry?", "extra error over clean encoding (px)"),
        (
            "excess_block_rel_agent_angle_rmse_deg",
            "How much do perturbations hurt T-to-agent orientation?",
            "extra error over clean encoding (degrees)",
        ),
        ("excess_objective_met_false_positive_rate", "How much do perturbations add false positives?", "extra false-positive rate over clean encoding"),
        ("excess_objective_met_recall", "How much do perturbations change success recall?", "recall change from clean encoding"),
    ]
    for metric, title, ylabel in excess_specs:
        if not any(metric in values for values in results.values()):
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        values = [results[label].get(metric, np.nan) for label in labels]
        ax.bar(labels, values)
        style_axis(ax, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0.0, color="0.25", linewidth=1.0, alpha=0.6)
        path = output_dir / "plots" / "excess" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    pair_specs = [
        ("agent_pos_rmse_px", "Agent Position Error", "RMSE px"),
        ("block_pos_rmse_px", "Block Position Error", "RMSE px"),
        ("block_angle_rmse_deg", "Block Angle Error", "RMSE deg"),
        ("block_rel_objective_xy_rmse_px", "T/Object XY Error", "RMSE px"),
        ("block_rel_objective_angle_rmse_deg", "T/Object Angle Error", "RMSE deg"),
        ("block_rel_agent_xy_rmse_px", "T/Agent XY Error", "RMSE px"),
        ("block_rel_agent_angle_rmse_deg", "T/Agent Angle Error", "RMSE deg"),
        ("objective_met_false_positive_rate", "Objective-Met False Positive Rate", "FPR"),
        ("objective_met_recall", "Objective-Met Recall", "recall"),
        ("objective_met_mean_probability", "Sparse Success Probability", "mean success probability"),
    ]
    normalized_specs = [
        ("agent_pos_rmse_px", "Agent Position", "clean-relative error increase"),
        ("block_pos_rmse_px", "Block Position", "clean-relative error increase"),
        ("block_angle_rmse_deg", "Block Angle", "clean-relative error increase"),
        ("block_rel_objective_xy_rmse_px", "T/Object XY", "clean-relative error increase"),
        ("block_rel_objective_angle_rmse_deg", "T/Object Angle", "clean-relative error increase"),
        ("block_rel_agent_xy_rmse_px", "T/Agent XY", "clean-relative error increase"),
        ("block_rel_agent_angle_rmse_deg", "T/Agent Angle", "clean-relative error increase"),
    ]
    x = np.arange(len(labels), dtype=np.float32)
    for suffix, title, ylabel in pair_specs:
        clean_key = f"clean_{suffix}"
        pert_key = f"perturbed_{suffix}"
        excess_key = f"excess_{suffix}"
        if not any(clean_key in values or pert_key in values for values in results.values()):
            continue
        clean_values = np.asarray([results[label].get(clean_key, np.nan) for label in labels], dtype=np.float32)
        pert_values = np.asarray([results[label].get(pert_key, np.nan) for label in labels], dtype=np.float32)
        excess_values = np.asarray(
            [
                results[label].get(excess_key, results[label].get(pert_key, np.nan) - results[label].get(clean_key, np.nan))
                for label in labels
            ],
            dtype=np.float32,
        )
        fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
        ax.bar(x, clean_values, color="0.78", edgecolor="0.35", linewidth=0.8, label="clean error")
        positive = np.where(excess_values > 0, excess_values, 0.0)
        negative = np.where(excess_values < 0, excess_values, 0.0)
        ax.bar(x, positive, bottom=clean_values, color="tab:blue", alpha=0.9, label="excess error")
        if np.any(negative < 0):
            ax.bar(x, negative, bottom=clean_values, color="tab:green", alpha=0.85, label="reduced error")
        style_axis(ax, title=f"Direct perturbation effect on {title.lower()} recovery", ylabel=ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        if "false_positive_rate" in suffix or "recall" in suffix or "probability" in suffix:
            ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False)
        path = output_dir / "plots" / "clean_vs_perturbed" / f"{suffix}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    for suffix, title, ylabel in normalized_specs:
        clean_key = f"clean_{suffix}"
        pert_key = f"perturbed_{suffix}"
        if not any(clean_key in values and pert_key in values for values in results.values()):
            continue
        clean_values = np.asarray([results[label].get(clean_key, np.nan) for label in labels], dtype=np.float32)
        pert_values = np.asarray([results[label].get(pert_key, np.nan) for label in labels], dtype=np.float32)
        degradation = np.divide(
            pert_values - clean_values,
            np.maximum(clean_values, 1e-8),
            out=np.full_like(clean_values, np.nan),
            where=np.isfinite(clean_values) & np.isfinite(pert_values),
        )
        fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
        colors = np.where(degradation >= 0.0, "tab:blue", "tab:green")
        ax.bar(x, 100.0 * degradation, color=colors, alpha=0.9)
        ax.axhline(0.0, color="0.25", linewidth=1.0, alpha=0.65)
        style_axis(ax, title=f"Percent loss in {title.lower()} recoverability", ylabel=f"{ylabel} (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        if suffix == "agent_pos_rmse_px":
            ax.set_ylim(bottom=0.0)
        path = output_dir / "plots" / "normalized_degradation" / f"{suffix}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_rollout_metric(output_dir: Path, env_steps: np.ndarray, results: dict[str, dict[str, np.ndarray]], metric: str, title: str, ylabel: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for perturbation, curves in results.items():
        if metric in curves:
            ax.plot(env_steps, curves[metric], linewidth=2, label=perturbation)
    style_axis(ax, title=title, xlabel="rollout horizon (environment steps)", ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=False)
    path = output_dir / "plots" / f"{metric}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollout_curves(output_dir: Path, env_steps: np.ndarray, results: dict[str, dict[str, np.ndarray]]) -> None:
    specs = [
        ("latent_cosine", "How much do perturbations alter rollout latent geometry?", "cosine similarity to clean imagined latent", (0.0, 1.0)),
        ("latent_rmse", "How far do perturbations move imagined latents?", "latent RMSE to clean imagined latent", None),
        ("excess_agent_pos_rmse_px", "Perturbation effect on agent-state recovery over time", "extra error over clean rollout (px)", None),
        ("excess_block_pos_rmse_px", "Perturbation effect on block-state recovery over time", "extra error over clean rollout (px)", None),
        ("excess_block_angle_rmse_deg", "Perturbation effect on block orientation over time", "extra error over clean rollout (degrees)", None),
        ("excess_block_rel_objective_xy_rmse_px", "Perturbation effect on T-to-goal geometry over time", "extra error over clean rollout (px)", None),
        ("excess_block_rel_agent_xy_rmse_px", "Perturbation effect on T-to-agent geometry over time", "extra error over clean rollout (px)", None),
        ("excess_objective_met_false_positive_rate", "Perturbation effect on sparse false positives over time", "extra false-positive rate over clean rollout", (-1.0, 1.0)),
        ("perturbed_objective_met_false_positive_rate", "Sparse false positives after perturbing context", "false-positive rate", (0.0, 1.05)),
        ("perturbed_objective_met_recall", "Sparse success recall after perturbing context", "recall", (0.0, 1.05)),
        ("perturbed_objective_met_mean_probability", "Sparse success probability after perturbing context", "mean success probability", (0.0, 1.0)),
    ]
    for metric, title, ylabel, ylim in specs:
        plot_rollout_metric(output_dir, env_steps, results, metric, title, ylabel, ylim)
    plot_rollout_normalized_degradation(output_dir, env_steps, results)


def plot_rollout_normalized_degradation(
    output_dir: Path,
    env_steps: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    specs = [
        ("agent_pos_rmse_px", "Agent Position", "extra error over clean rollout"),
        ("block_pos_rmse_px", "Block Position", "extra error over clean rollout"),
        ("block_angle_rmse_deg", "Block Angle", "extra error over clean rollout"),
        ("block_rel_objective_xy_rmse_px", "T/Object XY", "extra error over clean rollout"),
        ("block_rel_objective_angle_rmse_deg", "T/Object Angle", "extra error over clean rollout"),
        ("block_rel_agent_xy_rmse_px", "T/Agent XY", "extra error over clean rollout"),
        ("block_rel_agent_angle_rmse_deg", "T/Agent Angle", "extra error over clean rollout"),
    ]
    for suffix, title, ylabel in specs:
        clean_key = f"clean_{suffix}"
        pert_key = f"perturbed_{suffix}"
        if not any(clean_key in curves and pert_key in curves for curves in results.values()):
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for perturbation, curves in results.items():
            if clean_key not in curves or pert_key not in curves:
                continue
            clean = np.asarray(curves[clean_key], dtype=np.float32)
            perturbed = np.asarray(curves[pert_key], dtype=np.float32)
            degradation = np.divide(
                perturbed - clean,
                np.maximum(clean, 1e-8),
                out=np.full_like(clean, np.nan),
                where=np.isfinite(clean) & np.isfinite(perturbed),
            )
            ax.plot(env_steps, 100.0 * degradation, linewidth=2, label=perturbation)
        ax.axhline(0.0, color="0.25", linewidth=1.0, alpha=0.65)
        style_axis(
            ax,
            title=f"Percent loss in {title.lower()} recoverability over time",
            xlabel="rollout horizon (environment steps)",
            ylabel=f"{ylabel} (%)",
        )
        if suffix == "agent_pos_rmse_px":
            ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, frameon=False)
        path = output_dir / "plots" / "normalized_degradation" / f"{suffix}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_rollout_baseline_comparisons(
    output_dir: Path,
    env_steps: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    specs = [
        ("agent_pos_rmse_px", "How perturbations affect agent-state recovery during rollout", "agent position error (px)", None),
        ("block_pos_rmse_px", "How perturbations affect block-state recovery during rollout", "block position error (px)", None),
        ("block_angle_rmse_deg", "How perturbations affect block orientation during rollout", "block angle error (degrees)", None),
        ("block_rel_objective_xy_rmse_px", "How perturbations affect T-to-goal geometry during rollout", "T-to-goal position error (px)", None),
        ("block_rel_agent_xy_rmse_px", "How perturbations affect T-to-agent geometry during rollout", "T-to-agent position error (px)", None),
        ("objective_met_false_positive_rate", "Sparse false positives during perturbed rollouts", "false-positive rate", (0.0, 1.05)),
        ("objective_met_recall", "Sparse success recall during perturbed rollouts", "recall", (0.0, 1.05)),
        ("objective_met_mean_probability", "Sparse success probability during perturbed rollouts", "mean success probability", (0.0, 1.0)),
        (
            "latent_cosine_to_encoded_gt",
            "How much do perturbations alter latent geometry?",
            "cosine similarity to encoded ground truth",
            (0.0, 1.0),
        ),
        (
            "latent_rmse_to_encoded_gt",
            "How far do perturbations move imagined latents from ground truth?",
            "latent RMSE to encoded ground truth",
            None,
        ),
    ]
    if not results:
        return

    first_curves = next(iter(results.values()))
    for metric, title, ylabel, ylim in specs:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        encoded_key = f"encoded_gt_{metric}"
        clean_key = f"clean_{metric}"
        pert_key = f"perturbed_{metric}"
        if encoded_key in first_curves:
            ax.plot(env_steps, first_curves[encoded_key], color="black", linestyle="--", linewidth=2.5, label="encoded GT future")
        if clean_key in first_curves:
            ax.plot(env_steps, first_curves[clean_key], color="black", linewidth=2.0, alpha=0.75, label="clean imagined")
        for perturbation, curves in results.items():
            if pert_key in curves:
                ax.plot(env_steps, curves[pert_key], linewidth=1.7, label=f"{perturbation} imagined")
        style_axis(ax, title=title, xlabel="rollout horizon (environment steps)", ylabel=ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, ncols=2, frameon=False)
        path = output_dir / "plots" / "baseline_comparison" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)


def summarize_final(results: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {
        perturbation: {
            metric: float(values[-1])
            for metric, values in curves.items()
            if metric in (
                "latent_cosine",
                "latent_rmse",
                "excess_agent_pos_rmse_px",
                "excess_block_pos_rmse_px",
                "excess_block_angle_rmse_deg",
                "excess_block_rel_objective_xy_rmse_px",
                "excess_block_rel_agent_xy_rmse_px",
                "excess_objective_met_false_positive_rate",
                "perturbed_objective_met_false_positive_rate",
                "perturbed_objective_met_recall",
                "perturbed_objective_met_mean_probability",
            )
        }
        for perturbation, curves in results.items()
    }


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir or f"models/perturbation_probe/{Path(args.probe_dir).name}_{args.probe_kind}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_existing:
        plot_existing_outputs(output_dir, clean_plots=args.clean_plots)
        print(f"regenerated existing perturbation plots in {output_dir}")
        return

    device = torch.device(args.device)

    dataset_path = repo_path(args.dataset_path)
    checkpoint_cache_dir = repo_path(args.checkpoint_cache_dir)
    probe_dir = repo_path(args.probe_dir)

    probes, classifier_probes = probe_eval.load_probes(probe_dir, args.probe_kind)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=checkpoint_cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)

    summary: dict[str, object] = {"config": vars(args), "probe_dir": str(probe_dir)}
    with h5py.File(dataset_path, "r") as h5:
        plot_perturbation_examples(output_dir, h5, args)
        summary["perturbation_examples"] = str(output_dir / "perturbation_examples")
        if "direct" in args.analyses:
            direct_results, direct_arrays = evaluate_direct(args, h5, model, probes, classifier_probes, device)
            direct_dir = output_dir / "direct"
            write_direct_csv(direct_dir / "metrics.csv", direct_results)
            plot_direct_bars(direct_dir, direct_results)
            np.savez_compressed(direct_dir / "arrays.npz", **direct_arrays)
            summary["direct"] = direct_results
        if "rollout" in args.analyses:
            rollout_results, rollout_arrays = evaluate_rollout(args, h5, model, probes, classifier_probes, device)
            rollout_dir = output_dir / "rollout"
            env_steps = args.frameskip * np.arange(1, args.horizon + 1)
            write_rollout_csv(rollout_dir / "metrics.csv", env_steps, rollout_results)
            plot_rollout_curves(rollout_dir, env_steps, rollout_results)
            flat = {
                f"{perturbation}_{metric}": values
                for perturbation, curves in rollout_results.items()
                for metric, values in curves.items()
            }
            np.savez_compressed(rollout_dir / "arrays.npz", env_steps=env_steps, **rollout_arrays, **flat)
            summary["rollout_final_step"] = summarize_final(rollout_results)
        if "encoded_rollout" in args.analyses:
            rollout_dir = output_dir / "rollout"
            encoded_results, encoded_arrays = append_encoded_rollout_baseline(
                args,
                h5,
                model,
                probes,
                classifier_probes,
                device,
                rollout_dir,
            )
            env_steps = encoded_arrays["env_steps"]
            write_rollout_csv(rollout_dir / "metrics.csv", env_steps, encoded_results)
            plot_rollout_curves(rollout_dir, env_steps, encoded_results)
            plot_rollout_baseline_comparisons(rollout_dir, env_steps, encoded_results)
            np.savez_compressed(rollout_dir / "arrays.npz", **encoded_arrays)
            summary["encoded_rollout_final_step"] = summarize_final(encoded_results)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved perturbation invariance evaluation to {output_dir}")


if __name__ == "__main__":
    main()
