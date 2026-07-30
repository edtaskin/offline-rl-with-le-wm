"""Visualize PushT LeWM imagined rollouts with a trained latent image decoder.

This is the image-space companion to rollout_probe_pusht.py. It initializes LeWM
from ground-truth context frames, rolls forward with ground-truth action blocks,
decodes each imagined latent to an RGB frame, and saves:

  * side-by-side videos of real future frames vs decoded imagined frames,
  * a grid at model steps 1..10, i.e. environment steps 5..50 by default.
  * image-space rollout metrics and plots comparing decoded images.

Relative paths are resolved from the top-level wrapper repo, regardless of the
current working directory used to launch the script.
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
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder.train_decoder_pusht import LatentImageDecoder


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--decoder-checkpoint", default="models/latent_decoder/pusht_lewm/decoder_best.pt")
    parser.add_argument("--output-dir", default="models/rollout_decode/pusht_lewm")
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=3,
        help="Number of GT frames used to initialize rollout; 3 matches the trained PushT LeWM history.",
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--video-format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--grid-steps", default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--no-videos", action="store_true", help="Skip per-trajectory side-by-side videos.")
    parser.add_argument("--no-grids", action="store_true", help="Skip per-trajectory timestep grids.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--mask-dilate", type=int, default=7)
    parser.add_argument("--color-min", type=float, default=0.25)
    parser.add_argument("--color-margin", type=float, default=0.05)
    parser.add_argument("--gray-min", type=float, default=0.20)
    parser.add_argument("--gray-max", type=float, default=0.85)
    parser.add_argument("--gray-chroma", type=float, default=0.18)
    parser.add_argument("--min-mask-pixels", type=int, default=8)
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM.",
    )
    return parser.parse_args()


def preprocess_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 4, 2, 3).div_(255.0)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def sample_starts(
    h5: h5py.File,
    num_trajectories: int,
    context_steps: int,
    horizon: int,
    frameskip: int,
    seed: int,
) -> list[tuple[int, int]]:
    lengths = h5["ep_len"][:]
    offsets = h5["ep_offset"][:]
    required_last_frame = (context_steps + horizon - 1) * frameskip
    valid_eps = np.nonzero(lengths > required_last_frame)[0]
    if len(valid_eps) == 0:
        raise ValueError("No episodes are long enough for the requested context/horizon.")

    rng = np.random.default_rng(seed)
    ep_probs = lengths[valid_eps].astype(np.float64)
    ep_probs = ep_probs / ep_probs.sum()
    starts = []
    for _ in range(num_trajectories):
        ep = int(rng.choice(valid_eps, p=ep_probs))
        max_start = int(lengths[ep] - required_last_frame - 1)
        local_start = int(rng.integers(0, max_start + 1))
        starts.append((ep, int(offsets[ep] + local_start)))
    return starts


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:]
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def load_batch(
    h5: h5py.File,
    starts: list[tuple[int, int]],
    context_steps: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalize_actions: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz = len(starts)
    action_steps = context_steps + horizon - 1
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    future_pixels = np.empty((bsz, horizon, 224, 224, 3), dtype=np.uint8)
    action_blocks = np.empty((bsz, action_steps, frameskip * 2), dtype=np.float32)

    for i, (_, start) in enumerate(starts):
        context_idx = start + np.arange(context_steps) * frameskip
        future_idx = start + (context_steps + np.arange(horizon)) * frameskip
        context_pixels[i] = h5["pixels"][context_idx]
        future_pixels[i] = h5["pixels"][future_idx]
        for t in range(action_steps):
            a0 = start + t * frameskip
            raw_action = h5["action"][a0 : a0 + frameskip].astype(np.float32)
            if normalize_actions:
                raw_action = (raw_action - action_mean) / action_std
            action_blocks[i, t] = raw_action.reshape(-1)

    return context_pixels, future_pixels, action_blocks


@torch.inference_mode()
def rollout_embeddings(
    model: torch.nn.Module,
    context_pixels: np.ndarray,
    action_blocks: np.ndarray,
    horizon: int,
    history_size: int,
    device: torch.device,
) -> torch.Tensor:
    context = preprocess_pixels(context_pixels, device)
    actions = torch.from_numpy(action_blocks).to(device=device, dtype=torch.float32)

    init = model.encode({"pixels": context})
    emb_list = list(init["emb"].unbind(dim=1))
    all_act_emb = model.action_encoder(actions)
    context_steps = context.shape[1]

    for step in range(horizon):
        end = context_steps + step
        lo = max(0, end - history_size)
        emb_trunc = torch.stack(emb_list[lo:end], dim=1)
        act_trunc = all_act_emb[:, lo:end]
        pred = model.predict(emb_trunc, act_trunc)[:, -1]
        emb_list.append(pred)

    return torch.stack(emb_list[context_steps:], dim=1)


@torch.inference_mode()
def encode_future_embeddings(
    model: torch.nn.Module,
    future_pixels: np.ndarray,
    encode_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    bsz, horizon = future_pixels.shape[:2]
    flat_pixels = future_pixels.reshape(bsz * horizon, *future_pixels.shape[2:])
    chunks = []
    for start in range(0, len(flat_pixels), encode_batch_size):
        pixel_chunk = flat_pixels[start : start + encode_batch_size]
        pixels = preprocess_pixels(pixel_chunk[:, None], device)
        emb = model.encode({"pixels": pixels})["emb"][:, 0]
        chunks.append(emb.detach().cpu().float())
    return torch.cat(chunks, dim=0).reshape(bsz, horizon, -1)


def load_decoder(path: Path, device: torch.device) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing decoder checkpoint: {path}. Train one with scripts/decoder/train_decoder_pusht.py first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    decoder = LatentImageDecoder(**config)
    decoder.load_state_dict(payload["decoder"])
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder


@torch.inference_mode()
def decode_embeddings(
    decoder: nn.Module,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    bsz, horizon = embeddings.shape[:2]
    flat = embeddings.reshape(bsz * horizon, embeddings.shape[-1]).to(device=device, dtype=torch.float32)
    decoded = []
    for start in range(0, flat.size(0), batch_size):
        decoded.append(decoder(flat[start : start + batch_size]).detach().cpu())
    return torch.cat(decoded, dim=0).reshape(bsz, horizon, 3, 224, 224).clamp(0, 1)


def pixels_to_tensor(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels).permute(0, 1, 4, 2, 3).float().div(255.0)


def foreground_mask(images: torch.Tensor, threshold: float) -> torch.Tensor:
    return images.sub(1.0).abs().amax(dim=2) > threshold


def color_masks(images: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    r, g, b = images.unbind(dim=2)
    fg = foreground_mask(images, args.foreground_threshold)
    maxc = images.max(dim=2).values
    minc = images.min(dim=2).values
    mean = images.mean(dim=2)
    return {
        "foreground": fg,
        "blue_agent": (b > args.color_min) & (b > r + args.color_margin) & (b > g + args.color_margin),
        "green_target": (g > args.color_min) & (g > r + args.color_margin) & (g > b + args.color_margin),
        "gray_block": fg & ((maxc - minc) < args.gray_chroma) & (mean > args.gray_min) & (mean < args.gray_max),
    }


def _sum_over_images(values: torch.Tensor) -> np.ndarray:
    return values.sum(dim=(0, 2, 3)).detach().cpu().double().numpy()


def _sum_over_image_channels(values: torch.Tensor) -> np.ndarray:
    return values.sum(dim=(0, 2, 3, 4)).detach().cpu().double().numpy()


def _mask_den(mask: torch.Tensor, channels: int = 1) -> np.ndarray:
    return (mask.sum(dim=(0, 2, 3)) * channels).detach().cpu().double().numpy()


def ssim_maps(pred: torch.Tensor, ref: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred_flat = pred.flatten(0, 1)
    ref_flat = ref.flatten(0, 1)
    pad = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred_flat, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(ref_flat, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred_flat * pred_flat, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(ref_flat * ref_flat, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_flat * ref_flat, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.reshape(*pred.shape)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    flat = mask.flatten(0, 1).float().unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(flat, kernel_size=kernel_size, stride=1, padding=pad)
    return dilated.squeeze(1).reshape(mask.shape).bool()


def mask_iou_parts(pred_mask: torch.Tensor, ref_mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    intersection = torch.logical_and(pred_mask, ref_mask).sum(dim=(0, 2, 3))
    union = torch.logical_or(pred_mask, ref_mask).sum(dim=(0, 2, 3))
    return intersection.detach().cpu().double().numpy(), union.detach().cpu().double().numpy()


def centroid_error_parts(
    pred_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    min_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    bsz, horizon, height, width = pred_mask.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=pred_mask.device),
        torch.arange(width, dtype=torch.float32, device=pred_mask.device),
        indexing="ij",
    )
    numerator = torch.zeros(horizon, dtype=torch.float64)
    denominator = torch.zeros(horizon, dtype=torch.float64)
    pred_f = pred_mask.float()
    ref_f = ref_mask.float()

    for t in range(horizon):
        for b in range(bsz):
            pred_count = pred_f[b, t].sum()
            ref_count = ref_f[b, t].sum()
            if pred_count < min_pixels or ref_count < min_pixels:
                continue
            pred_x = (pred_f[b, t] * xs).sum() / pred_count
            pred_y = (pred_f[b, t] * ys).sum() / pred_count
            ref_x = (ref_f[b, t] * xs).sum() / ref_count
            ref_y = (ref_f[b, t] * ys).sum() / ref_count
            dist = torch.sqrt((pred_x - ref_x).pow(2) + (pred_y - ref_y).pow(2))
            numerator[t] += float(dist.detach().cpu())
            denominator[t] += 1.0
    return numerator.numpy(), denominator.numpy()


def image_metric_parts(
    pred: torch.Tensor,
    ref: torch.Tensor,
    prefix: str,
    args: argparse.Namespace,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pred = pred.detach().cpu().float().clamp(0, 1)
    ref = ref.detach().cpu().float().clamp(0, 1)
    bsz, horizon, channels, height, width = pred.shape
    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    abs_err = (pred - ref).abs()
    sq_err = (pred - ref).pow(2)
    full_den = np.full(horizon, bsz * channels * height * width, dtype=np.float64)
    parts[f"{prefix}_full_mae"] = (_sum_over_image_channels(abs_err), full_den)
    parts[f"{prefix}_full_rmse"] = (_sum_over_image_channels(sq_err), full_den)

    ref_fg = foreground_mask(ref, args.foreground_threshold)
    fg_den = _mask_den(ref_fg, channels)
    parts[f"{prefix}_foreground_mae"] = (_sum_over_image_channels(abs_err * ref_fg.unsqueeze(2)), fg_den)
    parts[f"{prefix}_foreground_rmse"] = (_sum_over_image_channels(sq_err * ref_fg.unsqueeze(2)), fg_den)

    ssim = ssim_maps(pred, ref)
    ssim_per_image = ssim.mean(dim=(2, 3, 4))
    parts[f"{prefix}_ssim"] = (
        ssim_per_image.sum(dim=0).detach().cpu().double().numpy(),
        np.full(horizon, bsz, dtype=np.float64),
    )
    fg_for_ssim = dilate_mask(ref_fg, args.mask_dilate)
    parts[f"{prefix}_foreground_ssim"] = (
        _sum_over_image_channels(ssim * fg_for_ssim.unsqueeze(2)),
        _mask_den(fg_for_ssim, channels),
    )

    pred_masks = color_masks(pred, args)
    ref_masks = color_masks(ref, args)
    for name in ("foreground", "blue_agent", "green_target", "gray_block"):
        inter, union = mask_iou_parts(pred_masks[name], ref_masks[name])
        parts[f"{prefix}_{name}_iou"] = (inter, union)
        num, den = centroid_error_parts(pred_masks[name], ref_masks[name], args.min_mask_pixels)
        parts[f"{prefix}_{name}_centroid_error_px"] = (num, den)

    return parts


class MetricAccumulator:
    def __init__(self):
        self.numerators: dict[str, np.ndarray] = {}
        self.denominators: dict[str, np.ndarray] = {}

    def add(self, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        for key, (num, den) in parts.items():
            if key not in self.numerators:
                self.numerators[key] = num.astype(np.float64).copy()
                self.denominators[key] = den.astype(np.float64).copy()
            else:
                self.numerators[key] += num
                self.denominators[key] += den

    def curves(self) -> dict[str, np.ndarray]:
        output = {}
        for key, num in self.numerators.items():
            den = self.denominators[key]
            value = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den > 0)
            if key.endswith("_rmse"):
                value = np.sqrt(value)
            output[key] = value
        return output


def save_metrics_csv(path: Path, env_steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_step", "env_step"] + sorted(curves)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, env_step in enumerate(env_steps):
            row = {"model_step": i + 1, "env_step": int(env_step)}
            row.update({key: float(value[i]) for key, value in curves.items()})
            writer.writerow(row)


def plot_metric_group(
    path: Path,
    env_steps: np.ndarray,
    curves: dict[str, np.ndarray],
    metric_names: list[tuple[str, str]],
    comparison_prefixes: list[tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4), constrained_layout=True)
    if len(metric_names) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metric_names):
        for prefix, label in comparison_prefixes:
            key = f"{prefix}_{metric}"
            if key in curves:
                ax.plot(env_steps, curves[key], label=label, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Environment steps after context")
        ax.grid(True, alpha=0.25)
    axes[-1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metric_plots(output_dir: Path, env_steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    comparisons = [
        ("imagined_vs_real", "imagined decoded vs real"),
        ("encoded_gt_vs_real", "encoded-GT decoded vs real"),
        ("imagined_vs_encoded_gt", "imagined decoded vs encoded-GT decoded"),
    ]
    plot_metric_group(
        output_dir / "rollout_decode_image_errors.png",
        env_steps,
        curves,
        [
            ("foreground_mae", "Foreground MAE"),
            ("foreground_rmse", "Foreground RMSE"),
            ("ssim", "SSIM"),
            ("foreground_ssim", "Foreground SSIM"),
        ],
        comparisons,
    )
    plot_metric_group(
        output_dir / "rollout_decode_mask_iou.png",
        env_steps,
        curves,
        [
            ("foreground_iou", "Foreground IoU"),
            ("blue_agent_iou", "Blue Agent IoU"),
            ("gray_block_iou", "Gray Block IoU"),
            ("green_target_iou", "Green Target IoU"),
        ],
        comparisons,
    )
    plot_metric_group(
        output_dir / "rollout_decode_centroid_errors.png",
        env_steps,
        curves,
        [
            ("foreground_centroid_error_px", "Foreground Centroid Error"),
            ("blue_agent_centroid_error_px", "Blue Agent Centroid Error"),
            ("gray_block_centroid_error_px", "Gray Block Centroid Error"),
            ("green_target_centroid_error_px", "Green Target Centroid Error"),
        ],
        comparisons,
    )


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def add_label(image: np.ndarray, label: str, header_height: int = 24) -> np.ndarray:
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + header_height), "white")
    canvas.paste(Image.fromarray(image), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return np.asarray(canvas)


def side_by_side_frame(real: torch.Tensor, decoded: torch.Tensor, env_step: int) -> np.ndarray:
    real_u8 = tensor_image_to_uint8(real)
    decoded_u8 = tensor_image_to_uint8(decoded)
    left = add_label(real_u8, f"real GT, +{env_step} env steps")
    right = add_label(decoded_u8, f"decoded imagined latent, +{env_step} env steps")
    divider = np.zeros((left.shape[0], 2, 3), dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def save_video(
    path: Path,
    real: torch.Tensor,
    decoded: torch.Tensor,
    env_steps: np.ndarray,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [side_by_side_frame(real[i], decoded[i], int(env_steps[i])) for i in range(len(env_steps))]
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps)
    else:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def parse_grid_steps(raw: str, frameskip: int, horizon: int) -> np.ndarray:
    env_steps = np.asarray([int(item.strip()) for item in raw.split(",") if item.strip()], dtype=np.int64)
    if len(env_steps) == 0:
        raise ValueError("--grid-steps must contain at least one environment step.")
    if np.any(env_steps % frameskip != 0):
        raise ValueError("--grid-steps must be multiples of --frameskip.")
    model_steps = env_steps // frameskip
    if np.any(model_steps < 1) or np.any(model_steps > horizon):
        raise ValueError("--grid-steps must be between one frameskip and horizon * frameskip.")
    return env_steps


def save_timestep_grid(
    path: Path,
    real: torch.Tensor,
    decoded: torch.Tensor,
    env_steps: np.ndarray,
    frameskip: int,
) -> None:
    selected = (env_steps // frameskip) - 1
    tile = 224
    left_margin = 70
    top_margin = 28
    pad = 6
    width = left_margin + len(selected) * tile + (len(selected) + 1) * pad
    height = top_margin + 2 * tile + 3 * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for col, idx in enumerate(selected):
        x = left_margin + pad + col * (tile + pad)
        draw.text((x + 6, 7), f"+{int(env_steps[col])}", fill=(0, 0, 0))
        y_real = top_margin + pad
        y_decoded = top_margin + 2 * pad + tile
        canvas.paste(Image.fromarray(tensor_image_to_uint8(real[int(idx)])), (x, y_real))
        canvas.paste(Image.fromarray(tensor_image_to_uint8(decoded[int(idx)])), (x, y_decoded))

    draw.text((8, top_margin + pad + tile // 2 - 8), "real", fill=(0, 0, 0))
    draw.text((8, top_margin + 2 * pad + tile + tile // 2 - 8), "decoded", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset_path = repo_path(args.dataset_path)
    checkpoint_cache_dir = repo_path(args.checkpoint_cache_dir)
    decoder_checkpoint = repo_path(args.decoder_checkpoint)

    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=checkpoint_cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", 3))
    decoder = load_decoder(decoder_checkpoint, device)
    grid_env_steps = parse_grid_steps(args.grid_steps, args.frameskip, args.horizon)
    env_steps = args.frameskip * np.arange(1, args.horizon + 1)
    metric_accumulator = MetricAccumulator()

    sampled: list[tuple[int, int]] = []
    with h5py.File(dataset_path, "r") as h5:
        action_mean, action_std = action_stats(h5)
        starts = sample_starts(
            h5=h5,
            num_trajectories=args.num_trajectories,
            context_steps=args.context_steps,
            horizon=args.horizon,
            frameskip=args.frameskip,
            seed=args.seed,
        )
        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start : batch_start + args.batch_size]
            context_pixels, future_pixels, action_blocks = load_batch(
                h5,
                batch_starts,
                context_steps=args.context_steps,
                horizon=args.horizon,
                frameskip=args.frameskip,
                action_mean=action_mean,
                action_std=action_std,
                normalize_actions=not args.no_normalize_actions,
            )
            pred_emb = rollout_embeddings(
                model,
                context_pixels,
                action_blocks,
                horizon=args.horizon,
                history_size=history_size,
                device=device,
            )
            decoded = decode_embeddings(decoder, pred_emb, args.batch_size, device)
            gt_emb = encode_future_embeddings(model, future_pixels, args.encode_batch_size, device)
            decoded_gt = decode_embeddings(decoder, gt_emb, args.batch_size, device)
            real = pixels_to_tensor(future_pixels)

            metric_accumulator.add(image_metric_parts(decoded, real, "imagined_vs_real", args))
            metric_accumulator.add(image_metric_parts(decoded_gt, real, "encoded_gt_vs_real", args))
            metric_accumulator.add(
                image_metric_parts(decoded, decoded_gt, "imagined_vs_encoded_gt", args)
            )

            for i, (ep, start) in enumerate(batch_starts):
                traj_name = f"trajectory_{len(sampled):03d}_ep{ep}_start{start}"
                suffix = ".gif" if args.video_format == "gif" else ".mp4"
                if not args.no_videos:
                    save_video(
                        output_dir / f"{traj_name}_side_by_side{suffix}",
                        real[i],
                        decoded[i],
                        env_steps,
                        args.fps,
                    )
                if not args.no_grids:
                    save_timestep_grid(
                        output_dir / f"{traj_name}_grid.png",
                        real[i],
                        decoded[i],
                        grid_env_steps,
                        args.frameskip,
                    )
                sampled.append((ep, start))
                if args.no_videos and args.no_grids:
                    print(f"processed metrics for {traj_name}")
                else:
                    print(f"saved visuals for {traj_name}")

    metric_curves = metric_accumulator.curves()
    save_metrics_csv(output_dir / "rollout_decode_metrics.csv", env_steps, metric_curves)
    save_metric_plots(output_dir, env_steps, metric_curves)
    np.savez_compressed(
        output_dir / "rollout_decode_metric_arrays.npz",
        env_steps=env_steps,
        **metric_curves,
    )

    summary = {
        "config": vars(args),
        "num_trajectories": len(sampled),
        "sampled": [{"episode": int(ep), "start": int(start)} for ep, start in sampled],
        "history_size": history_size,
        "env_steps": (args.frameskip * np.arange(1, args.horizon + 1)).tolist(),
        "grid_env_steps": grid_env_steps.tolist(),
        "final_step": {key: float(value[-1]) for key, value in metric_curves.items()},
        "mean_over_horizon": {key: float(np.nanmean(value)) for key, value in metric_curves.items()},
        "action_normalization": {
            "enabled": not args.no_normalize_actions,
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
    }
    with (output_dir / "rollout_decode_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
