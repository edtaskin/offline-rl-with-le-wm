"""Train augmentation-invariant MLP probes on frozen PushT LeWM latents.

Each training sample is encoded twice:
  - clean frame -> z_clean
  - perturbed copy of the same frame -> z_aug

The probe predicts the same physical target from both latents and is penalized
when the two predictions disagree. LeWM stays frozen. The saved checkpoints use
the same format as ``train_probes_pusht.py`` MLP probes, so rollout evaluators
can load them with ``--probe-kind mlp``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.train_probes_pusht import (  # noqa: E402
    ANGLE_FEATURES,
    CLASSIFICATION_FEATURES,
    FEATURES,
    RELATIVE_POSE_FEATURES,
    SplitConfig,
    binary_confusion,
    classification_metrics,
    dataset_path,
    evaluate,
    load_latent_cache,
    pearsonr,
    preprocess_pixels,
    repo_path,
    sample_rows,
    save_latent_cache,
    standardize,
    targets,
)


PERTURBATION_CHOICES = (
    "agent_color",
    "block_color",
    "target_color",
    "brightness",
    "gaussian_noise",
    "blur",
)
PERTURBATION_IMPL_VERSION = 4
DEFAULT_PERTURBATIONS = ["agent_color", "block_color", "target_color", "brightness", "gaussian_noise", "blur"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/probes/pusht_lewm_invariant")
    parser.add_argument("--clean-latent-cache", default=None)
    parser.add_argument("--paired-latent-cache", default=None)
    parser.add_argument("--rollout-paired-latent-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--rollout-max-samples", type=int, default=None)
    parser.add_argument("--sample-block-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument(
        "--training-sources",
        choices=["direct", "rollout", "direct_and_rollout"],
        default="direct",
        help="Train on direct encoded frame pairs, rollout latent pairs, or both.",
    )
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization before feeding GT actions to LeWM rollouts.",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["agent_pos", "block_pos", "block_angle", "block_rel_objective", "objective_met"],
        choices=["all", *FEATURES.keys()],
        help="Probe names to train. Use 'all' to train every supported probe.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=DEFAULT_PERTURBATIONS,
        choices=PERTURBATION_CHOICES,
    )
    parser.add_argument("--agent-color", nargs=3, type=int, default=[220, 40, 40])
    parser.add_argument("--block-color", nargs=3, type=int, default=[150, 90, 220])
    parser.add_argument("--target-color", nargs=3, type=int, default=[255, 185, 60])
    parser.add_argument("--brightness-delta", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=12.0)
    parser.add_argument("--blur-radius", type=float, default=1.25)
    parser.add_argument("--color-min", type=float, default=0.25)
    parser.add_argument("--color-margin", type=float, default=0.05)
    parser.add_argument("--gray-min", type=float, default=0.20)
    parser.add_argument("--gray-max", type=float, default=0.85)
    parser.add_argument("--gray-chroma", type=float, default=0.18)
    parser.add_argument(
        "--mask-border-margin",
        type=int,
        default=8,
        help="Ignore this many edge pixels for color-object masks to avoid recoloring the arena border.",
    )
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    return parser.parse_args()


def selected_probes(args: argparse.Namespace) -> list[str]:
    if "all" in args.probes:
        return list(FEATURES.keys())
    return list(dict.fromkeys(args.probes))


def color_masks(pixels: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    x = pixels.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maxc = x.max(axis=-1)
    minc = x.min(axis=-1)
    chroma = maxc - minc
    h, w = pixels.shape[:2]
    margin = int(getattr(args, "mask_border_margin", 8))
    yy, xx = np.ogrid[:h, :w]
    interior = (yy >= margin) & (yy < h - margin) & (xx >= margin) & (xx < w - margin)
    agent_margin = max(float(args.color_margin), 0.12)
    target_margin = max(float(args.color_margin), 0.10)
    agent = interior & (b > 0.45) & (b > r + agent_margin) & (b > g + agent_margin) & (chroma > 0.18)
    target = interior & (g > args.color_min) & (g > r + target_margin) & (g > b + target_margin)
    block = interior & (maxc > args.gray_min) & (maxc < args.gray_max) & (chroma < args.gray_chroma) & ~agent & ~target
    return {
        "agent": agent,
        "target": target,
        "block": block,
    }


def apply_perturbation(
    pixels: np.ndarray,
    perturbation: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    out = pixels.copy()
    masks = color_masks(out, args)
    if perturbation == "agent_color":
        out[masks["agent"]] = np.asarray(args.agent_color, dtype=np.uint8)
    elif perturbation == "block_color":
        out[masks["block"]] = np.asarray(args.block_color, dtype=np.uint8)
    elif perturbation == "target_color":
        out[masks["target"]] = np.asarray(args.target_color, dtype=np.uint8)
    elif perturbation == "brightness":
        delta = int(round(255.0 * args.brightness_delta))
        out = np.clip(out.astype(np.int16) - delta, 0, 255).astype(np.uint8)
    elif perturbation == "gaussian_noise":
        noise = rng.normal(0.0, args.noise_std, size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif perturbation == "blur":
        out = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=args.blur_radius)))
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")
    return out


def perturb_batch(
    pixels: np.ndarray,
    perturbations: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    aug = np.empty_like(pixels)
    ids = np.empty((len(pixels),), dtype=np.int64)
    for i, frame in enumerate(pixels):
        ids[i] = int(rng.integers(0, len(perturbations)))
        aug[i] = apply_perturbation(frame, perturbations[ids[i]], args, rng)
    return aug, ids


def contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    breaks = np.nonzero(np.diff(rows) != 1)[0] + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(rows, breaks)]


def iter_h5_batches(h5_file: h5py.File, rows: np.ndarray, batch_size: int):
    pixels_parts = []
    state_parts = []
    row_parts = []
    n_buffered = 0
    for run_start, run_end in contiguous_runs(rows):
        for start in range(run_start, run_end, batch_size):
            end = min(start + batch_size, run_end)
            pixels_parts.append(h5_file["pixels"][start:end])
            state_parts.append(h5_file["state"][start:end].astype(np.float32))
            row_parts.append(np.arange(start, end, dtype=np.int64))
            n_buffered += end - start
            if n_buffered >= batch_size:
                pixels = np.concatenate(pixels_parts, axis=0)
                states = np.concatenate(state_parts, axis=0)
                batch_rows = np.concatenate(row_parts, axis=0)
                yield pixels[:batch_size], states[:batch_size], batch_rows[:batch_size]
                pixels_parts = [pixels[batch_size:]] if len(pixels[batch_size:]) else []
                state_parts = [states[batch_size:]] if len(states[batch_size:]) else []
                row_parts = [batch_rows[batch_size:]] if len(batch_rows[batch_size:]) else []
                n_buffered = len(pixels_parts[0]) if pixels_parts else 0
    if n_buffered:
        yield np.concatenate(pixels_parts, axis=0), np.concatenate(state_parts, axis=0), np.concatenate(row_parts, axis=0)


@torch.inference_mode()
def encode_paired_rows(
    model: nn.Module,
    h5_path: Path,
    rows_by_split: dict[str, np.ndarray],
    batch_size: int,
    perturbations: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray]]:
    device = torch.device(args.device)
    model.eval().to(device)
    output: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(args.seed + 2000)

    with h5py.File(h5_path, "r") as f:
        for split, rows in rows_by_split.items():
            clean_parts = []
            aug_parts = []
            state_parts = []
            row_parts = []
            perturb_parts = []
            for batch_idx, (pixel_batch, state_batch, row_batch) in enumerate(iter_h5_batches(f, rows, batch_size)):
                aug_batch, perturb_ids = perturb_batch(pixel_batch, perturbations, args, rng)
                clean_pixels = preprocess_pixels(pixel_batch, device)
                aug_pixels = preprocess_pixels(aug_batch, device)
                clean_emb = model.encode({"pixels": clean_pixels.unsqueeze(1)})["emb"][:, 0]
                aug_emb = model.encode({"pixels": aug_pixels.unsqueeze(1)})["emb"][:, 0]
                clean_parts.append(clean_emb.detach().cpu().float().numpy())
                aug_parts.append(aug_emb.detach().cpu().float().numpy())
                state_parts.append(state_batch)
                row_parts.append(row_batch)
                perturb_parts.append(perturb_ids)
                if (batch_idx + 1) % 25 == 0:
                    done = min((batch_idx + 1) * batch_size, len(rows))
                    print(f"encoding paired {split}: {done}/{len(rows)}")

            output[split] = {
                "z_clean": np.concatenate(clean_parts, axis=0).astype(np.float32),
                "z_aug": np.concatenate(aug_parts, axis=0).astype(np.float32),
                "state": np.concatenate(state_parts, axis=0).astype(np.float32),
                "rows": np.concatenate(row_parts, axis=0).astype(np.int64),
                "perturbation_id": np.concatenate(perturb_parts, axis=0).astype(np.int64),
            }
            print(f"cached paired {split}: {len(rows)} samples")
    return output


def save_paired_cache(path: Path, data: dict[str, dict[str, np.ndarray]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": np.array(json.dumps(metadata))}
    for split, split_data in data.items():
        for key, value in split_data.items():
            payload[f"{split}_{key}"] = value
    np.savez_compressed(path, **payload)


def load_paired_cache(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    raw = np.load(path, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    data = {}
    for split in ("train", "val", "test"):
        data[split] = {
            "z_clean": raw[f"{split}_z_clean"],
            "z_aug": raw[f"{split}_z_aug"],
            "state": raw[f"{split}_state"],
            "rows": raw[f"{split}_rows"],
            "perturbation_id": raw[f"{split}_perturbation_id"],
        }
    return data, metadata


def paired_cache_matches(found: dict, expected: dict) -> bool:
    return all(found.get(key) == expected.get(key) for key in expected)


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:]
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def sample_rollout_starts(
    offsets: np.ndarray,
    lengths: np.ndarray,
    source_rows: np.ndarray,
    num_windows: int,
    context_steps: int,
    horizon: int,
    frameskip: int,
    seed: int,
) -> list[tuple[int, int]]:
    source_episodes = np.unique(np.searchsorted(offsets, source_rows, side="right") - 1)
    required_last_frame = (context_steps + horizon - 1) * frameskip
    valid_eps = source_episodes[lengths[source_episodes] > required_last_frame]
    if len(valid_eps) == 0:
        raise ValueError("No split episodes are long enough for rollout-pair generation.")

    rng = np.random.default_rng(seed)
    ep_probs = lengths[valid_eps].astype(np.float64)
    ep_probs = ep_probs / ep_probs.sum()
    starts = []
    for _ in range(num_windows):
        ep = int(rng.choice(valid_eps, p=ep_probs))
        max_start = int(lengths[ep] - required_last_frame - 1)
        local_start = int(rng.integers(0, max_start + 1))
        starts.append((ep, int(offsets[ep] + local_start)))
    return starts


def load_rollout_pair_batch(
    h5: h5py.File,
    starts: list[tuple[int, int]],
    context_steps: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalize_actions: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bsz = len(starts)
    action_steps = context_steps + horizon - 1
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    action_blocks = np.empty((bsz, action_steps, frameskip * 2), dtype=np.float32)
    future_rows = np.empty((bsz, horizon), dtype=np.int64)
    future_states = np.empty((bsz, horizon, h5["state"].shape[1]), dtype=np.float32)
    episodes = np.empty((bsz,), dtype=np.int64)

    for i, (ep, start) in enumerate(starts):
        episodes[i] = ep
        context_idx = start + np.arange(context_steps) * frameskip
        future_idx = start + (context_steps + np.arange(horizon)) * frameskip
        context_pixels[i] = h5["pixels"][context_idx]
        future_rows[i] = future_idx
        future_states[i] = h5["state"][future_idx].astype(np.float32)
        for t in range(action_steps):
            a0 = start + t * frameskip
            raw_action = h5["action"][a0 : a0 + frameskip].astype(np.float32)
            if normalize_actions:
                raw_action = (raw_action - action_mean) / action_std
            action_blocks[i, t] = raw_action.reshape(-1)
    return context_pixels, action_blocks, future_rows, future_states, episodes


def perturb_context_batch(
    context_pixels: np.ndarray,
    perturbations: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bsz, context_steps = context_pixels.shape[:2]
    flat_aug, flat_ids = perturb_batch(context_pixels.reshape(bsz * context_steps, *context_pixels.shape[2:]), perturbations, args, rng)
    return flat_aug.reshape(context_pixels.shape), flat_ids.reshape(bsz, context_steps)


def preprocess_context_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 4, 2, 3).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def rollout_embeddings(
    model: nn.Module,
    context_pixels: np.ndarray,
    action_blocks: np.ndarray,
    horizon: int,
    history_size: int,
    device: torch.device,
) -> torch.Tensor:
    context = preprocess_context_pixels(context_pixels, device)
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
    return torch.stack(emb_list[context_steps:], dim=1).detach().cpu()


@torch.inference_mode()
def encode_rollout_paired_rows(
    model: nn.Module,
    h5_path: Path,
    rows_by_split: dict[str, np.ndarray],
    num_samples_by_split: dict[str, int],
    batch_size: int,
    perturbations: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray]]:
    device = torch.device(args.device)
    model.eval().to(device)
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))
    rng = np.random.default_rng(args.seed + 4000)
    output: dict[str, dict[str, np.ndarray]] = {}

    with h5py.File(h5_path, "r") as h5:
        offsets = h5["ep_offset"][:].astype(np.int64)
        lengths = h5["ep_len"][:].astype(np.int64)
        action_mean, action_std = action_stats(h5)
        for split, source_rows in rows_by_split.items():
            num_samples = int(num_samples_by_split[split])
            starts = sample_rollout_starts(
                offsets,
                lengths,
                source_rows,
                num_windows=int(np.ceil(num_samples / args.rollout_horizon)),
                context_steps=args.context_steps,
                horizon=args.rollout_horizon,
                frameskip=args.frameskip,
                seed=args.seed + 5000 + {"train": 0, "val": 1, "test": 2}[split],
            )
            clean_parts = []
            aug_parts = []
            state_parts = []
            row_parts = []
            episode_parts = []
            model_step_parts = []
            start_parts = []
            perturb_parts = []

            for start_idx in range(0, len(starts), batch_size):
                batch_starts = starts[start_idx : start_idx + batch_size]
                context_pixels, action_blocks, future_rows, future_states, episodes = load_rollout_pair_batch(
                    h5,
                    batch_starts,
                    context_steps=args.context_steps,
                    horizon=args.rollout_horizon,
                    frameskip=args.frameskip,
                    action_mean=action_mean,
                    action_std=action_std,
                    normalize_actions=not args.no_normalize_actions,
                )
                aug_context, perturb_ids = perturb_context_batch(context_pixels, perturbations, args, rng)
                clean_emb = rollout_embeddings(model, context_pixels, action_blocks, args.rollout_horizon, history_size, device)
                aug_emb = rollout_embeddings(model, aug_context, action_blocks, args.rollout_horizon, history_size, device)

                clean_parts.append(clean_emb.reshape(-1, clean_emb.shape[-1]).numpy())
                aug_parts.append(aug_emb.reshape(-1, aug_emb.shape[-1]).numpy())
                state_parts.append(future_states.reshape(-1, future_states.shape[-1]))
                row_parts.append(future_rows.reshape(-1))
                episode_parts.append(np.repeat(episodes, args.rollout_horizon))
                model_step_parts.append(np.tile(np.arange(1, args.rollout_horizon + 1, dtype=np.int64), len(episodes)))
                start_parts.append(np.repeat(np.asarray([s for _, s in batch_starts], dtype=np.int64), args.rollout_horizon))
                perturb_parts.append(np.repeat(perturb_ids[:, -1], args.rollout_horizon))
                done = min(sum(len(part) for part in row_parts), num_samples)
                print(f"encoding rollout paired {split}: {done}/{num_samples}")

            output[split] = {
                "z_clean": np.concatenate(clean_parts, axis=0)[:num_samples].astype(np.float32),
                "z_aug": np.concatenate(aug_parts, axis=0)[:num_samples].astype(np.float32),
                "state": np.concatenate(state_parts, axis=0)[:num_samples].astype(np.float32),
                "rows": np.concatenate(row_parts, axis=0)[:num_samples].astype(np.int64),
                "perturbation_id": np.concatenate(perturb_parts, axis=0)[:num_samples].astype(np.int64),
                "episode": np.concatenate(episode_parts, axis=0)[:num_samples].astype(np.int64),
                "model_step": np.concatenate(model_step_parts, axis=0)[:num_samples].astype(np.int64),
                "start": np.concatenate(start_parts, axis=0)[:num_samples].astype(np.int64),
                "history_size": np.asarray([history_size], dtype=np.int64),
            }
            print(f"cached rollout paired {split}: {len(output[split]['rows'])} samples")
    return output


def merge_pair_sources(
    direct: dict[str, dict[str, np.ndarray]] | None,
    rollout: dict[str, dict[str, np.ndarray]] | None,
    source_mode: str,
) -> dict[str, dict[str, np.ndarray]]:
    if source_mode == "direct":
        if direct is None:
            raise ValueError("direct pair data is required for --training-sources direct")
        for split in ("train", "val", "test"):
            direct[split]["source_id"] = np.zeros((len(direct[split]["rows"]),), dtype=np.int64)
        return direct
    if source_mode == "rollout":
        if rollout is None:
            raise ValueError("rollout pair data is required for --training-sources rollout")
        for split in ("train", "val", "test"):
            rollout[split]["source_id"] = np.ones((len(rollout[split]["rows"]),), dtype=np.int64)
        return rollout
    if direct is None or rollout is None:
        raise ValueError("both direct and rollout pair data are required for --training-sources direct_and_rollout")

    merged = {}
    for split in ("train", "val", "test"):
        merged[split] = {}
        for key in ("z_clean", "z_aug", "state", "rows", "perturbation_id"):
            merged[split][key] = np.concatenate([direct[split][key], rollout[split][key]], axis=0)
        merged[split]["source_id"] = np.concatenate(
            [
                np.zeros((len(direct[split]["rows"]),), dtype=np.int64),
                np.ones((len(rollout[split]["rows"]),), dtype=np.int64),
            ],
            axis=0,
        )
    return merged


def split_sample_counts(total: int, split_cfg: SplitConfig, rows_by_split: dict[str, np.ndarray]) -> dict[str, int]:
    if total < 0:
        return {split: int(len(rows)) for split, rows in rows_by_split.items()}
    train = int(total * split_cfg.train_fraction)
    val = int(total * split_cfg.val_fraction)
    test = int(total - train - val)
    return {"train": train, "val": val, "test": test}


def rows_fingerprint(rows: np.ndarray) -> dict[str, int | str]:
    rows = np.asarray(rows, dtype=np.int64)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    return {
        "count": int(len(rows)),
        "first": int(rows[0]) if len(rows) else -1,
        "last": int(rows[-1]) if len(rows) else -1,
        "sha256": digest,
    }


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, classification: bool) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).float()
        out = model(xb)
        if classification:
            out = torch.sigmoid(out)
        preds.append(out.numpy())
    return np.concatenate(preds, axis=0)


def train_invariant_mlp(
    x_clean: np.ndarray,
    x_aug: np.ndarray,
    y_train: np.ndarray,
    x_val_clean: np.ndarray,
    x_val_aug: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    *,
    classification: bool,
) -> ProbeMLP:
    device = torch.device(args.device)
    model = ProbeMLP(x_clean.shape[1], y_train.shape[1], args.mlp_hidden, args.mlp_depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if classification:
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        if pos <= 0:
            raise ValueError("classification target has no positive examples in the training split")
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        supervised_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        supervised_loss = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_clean).float(),
        torch.from_numpy(x_aug).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    xvc = torch.from_numpy(x_val_clean).float().to(device)
    xva = torch.from_numpy(x_val_aug).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)

    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for clean_b, aug_b, yb in loader:
            clean_b = clean_b.to(device)
            aug_b = aug_b.to(device)
            yb = yb.to(device)
            pred_clean = model(clean_b)
            pred_aug = model(aug_b)
            if classification:
                sup = supervised_loss(pred_clean, yb) + supervised_loss(pred_aug, yb)
                cons = nn.functional.mse_loss(torch.sigmoid(pred_clean), torch.sigmoid(pred_aug))
            else:
                sup = supervised_loss(pred_clean, yb) + supervised_loss(pred_aug, yb)
                cons = nn.functional.mse_loss(pred_clean, pred_aug)
            loss = 0.5 * sup + args.consistency_weight * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            val_clean = model(xvc)
            val_aug = model(xva)
            if classification:
                val_sup = supervised_loss(val_clean, yv) + supervised_loss(val_aug, yv)
                val_cons = nn.functional.mse_loss(torch.sigmoid(val_clean), torch.sigmoid(val_aug))
            else:
                val_sup = supervised_loss(val_clean, yv) + supervised_loss(val_aug, yv)
                val_cons = nn.functional.mse_loss(val_clean, val_aug)
            val_loss = (0.5 * val_sup + args.consistency_weight * val_cons).item()
        print(f"epoch {epoch:03d} val_loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.reshape(-1).astype(np.float32)
    p = y_prob.reshape(-1)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    candidates = np.concatenate([[0.5], candidates])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = binary_confusion(y, p, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "pearson": pearsonr(pred, true),
    }


def prediction_consistency(pred_clean: np.ndarray, pred_aug: np.ndarray, classification: bool) -> dict[str, float]:
    if classification:
        return {
            "prob_mae": float(np.mean(np.abs(pred_clean - pred_aug))),
            "prob_rmse": float(np.sqrt(np.mean((pred_clean - pred_aug) ** 2))),
        }
    return regression_metrics(pred_clean, pred_aug)


def train_all(data: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> dict:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train_clean, (x_val_clean, x_test_clean, x_train_aug, x_val_aug, x_test_aug), x_mean, x_std = standardize(
        data["train"]["z_clean"].astype(np.float32),
        data["val"]["z_clean"].astype(np.float32),
        data["test"]["z_clean"].astype(np.float32),
        data["train"]["z_aug"].astype(np.float32),
        data["val"]["z_aug"].astype(np.float32),
        data["test"]["z_aug"].astype(np.float32),
    )

    results = {
        "config": vars(args),
        "normalizer": {"mean": x_mean.tolist(), "std": x_std.tolist()},
        "sample_counts": {
            split: {
                "total": int(len(split_data["rows"])),
                "direct": int((split_data.get("source_id", np.zeros(len(split_data["rows"]), dtype=np.int64)) == 0).sum()),
                "rollout": int((split_data.get("source_id", np.zeros(len(split_data["rows"]), dtype=np.int64)) == 1).sum()),
            }
            for split, split_data in data.items()
        },
        "probes": {},
    }

    for feature in selected_probes(args):
        feature_dir = output_dir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)
        classification = feature in CLASSIFICATION_FEATURES
        y_train_raw = targets(data["train"]["state"], feature, args)
        y_val_raw = targets(data["val"]["state"], feature, args)
        y_test_raw = targets(data["test"]["state"], feature, args)
        if classification or feature in ANGLE_FEATURES:
            y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw
            y_mean = np.zeros((1, y_train.shape[1]), dtype=np.float32)
            y_std = np.ones((1, y_train.shape[1]), dtype=np.float32)
        else:
            y_train, (y_val, y_test), y_mean, y_std = standardize(y_train_raw, y_val_raw, y_test_raw)

        print(f"training invariant MLP for {feature}")
        model = train_invariant_mlp(
            x_train_clean,
            x_train_aug,
            y_train,
            x_val_clean,
            x_val_aug,
            y_val,
            args,
            classification=classification,
        )

        if classification:
            val_clean_prob = predict(model, x_val_clean, args.batch_size, classification=True)
            threshold = select_threshold(y_val, val_clean_prob)
            test_clean_prob = predict(model, x_test_clean, args.batch_size, classification=True)
            test_aug_prob = predict(model, x_test_aug, args.batch_size, classification=True)
            clean_metrics = classification_metrics(test_clean_prob, y_test, threshold)
            aug_metrics = classification_metrics(test_aug_prob, y_test, threshold)
            consistency = prediction_consistency(test_clean_prob, test_aug_prob, classification=True)
            save_payload = {
                "model": model.state_dict(),
                "input_dim": x_train_clean.shape[1],
                "output_dim": y_train.shape[1],
                "hidden_dim": args.mlp_hidden,
                "depth": args.mlp_depth,
                "x_mean": x_mean,
                "x_std": x_std,
                "threshold": threshold,
                "task": "binary_classification",
                "invariant_training": True,
                "consistency_weight": args.consistency_weight,
            }
        else:
            test_clean_pred = predict(model, x_test_clean, args.batch_size, classification=False) * y_std + y_mean
            test_aug_pred = predict(model, x_test_aug, args.batch_size, classification=False) * y_std + y_mean
            clean_metrics = evaluate(feature, test_clean_pred, y_test_raw)
            aug_metrics = evaluate(feature, test_aug_pred, y_test_raw)
            consistency = prediction_consistency(test_clean_pred, test_aug_pred, classification=False)
            save_payload = {
                "model": model.state_dict(),
                "input_dim": x_train_clean.shape[1],
                "output_dim": y_train.shape[1],
                "hidden_dim": args.mlp_hidden,
                "depth": args.mlp_depth,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "invariant_training": True,
                "consistency_weight": args.consistency_weight,
            }

        torch.save(save_payload, feature_dir / "mlp_probe.pt")
        results["probes"][feature] = {
            "clean_test": clean_metrics,
            "perturbed_test": aug_metrics,
            "clean_vs_perturbed_prediction_consistency": consistency,
        }
        print(f"{feature} clean:     {clean_metrics}")
        print(f"{feature} perturbed: {aug_metrics}")
        print(f"{feature} consistency: {consistency}")

    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    h5_path = dataset_path(args.cache_dir, args.dataset)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo_path(args.cache_dir)
    split_cfg = SplitConfig(seed=args.seed)
    clean_cache = repo_path(args.clean_latent_cache) if args.clean_latent_cache else output_dir / "clean_latents.npz"
    paired_cache = repo_path(args.paired_latent_cache) if args.paired_latent_cache else output_dir / "paired_latents.npz"
    rollout_paired_cache = (
        repo_path(args.rollout_paired_latent_cache)
        if args.rollout_paired_latent_cache
        else output_dir / "rollout_paired_latents.npz"
    )
    needs_direct = args.training_sources in ("direct", "direct_and_rollout")
    needs_rollout = args.training_sources in ("rollout", "direct_and_rollout")

    if clean_cache.exists() and not args.force_recache:
        clean_data, clean_metadata = load_latent_cache(clean_cache)
        rows_by_split = {split: split_data["rows"] for split, split_data in clean_data.items()}
        print(f"loaded clean latent cache rows: {clean_cache}")
    elif needs_direct:
        rows_by_split = sample_rows(h5_path, args.max_samples, args.sample_block_size, split_cfg)
        clean_metadata = {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "cache_dir": str(cache_dir),
            "max_samples": args.max_samples,
            "sample_block_size": args.sample_block_size,
            "split": asdict(split_cfg),
        }
        model_for_clean = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        clean_data = {}
        # Reuse the normal cache format so this can also seed regular probe training.
        from scripts.probes.train_probes_pusht import encode_rows

        clean_data = encode_rows(
            model=model_for_clean,
            h5_path=h5_path,
            rows_by_split=rows_by_split,
            batch_size=args.encode_batch_size,
            device=torch.device(args.device),
        )
        save_latent_cache(clean_cache, clean_data, clean_metadata)
        print(f"saved clean latent cache: {clean_cache}")
    else:
        rows_by_split = sample_rows(h5_path, args.max_samples, args.sample_block_size, split_cfg)
        clean_metadata = {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "cache_dir": str(cache_dir),
            "max_samples": args.max_samples,
            "sample_block_size": args.sample_block_size,
            "split": asdict(split_cfg),
            "rows_only": True,
        }

    paired_expected = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "sample_block_size": args.sample_block_size,
        "split": asdict(split_cfg),
        "rows": {split: rows_fingerprint(rows) for split, rows in rows_by_split.items()},
        "perturbation_impl_version": PERTURBATION_IMPL_VERSION,
        "perturbations": list(args.perturbations),
        "agent_color": list(args.agent_color),
        "block_color": list(args.block_color),
        "target_color": list(args.target_color),
        "brightness_delta": args.brightness_delta,
        "noise_std": args.noise_std,
        "blur_radius": args.blur_radius,
        "mask_border_margin": args.mask_border_margin,
        "seed": args.seed,
    }
    direct_data = None
    paired_metadata = None
    if needs_direct:
        if paired_cache.exists() and not args.force_recache:
            direct_data, paired_metadata = load_paired_cache(paired_cache)
            if paired_cache_matches(paired_metadata, paired_expected):
                print(f"loaded paired latent cache: {paired_cache}")
            else:
                print(f"paired cache metadata mismatch, recaching: {paired_cache}")
                model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
                direct_data = encode_paired_rows(
                    model,
                    h5_path,
                    rows_by_split,
                    args.encode_batch_size,
                    list(args.perturbations),
                    args,
                )
                paired_metadata = paired_expected
                save_paired_cache(paired_cache, direct_data, paired_metadata)
                print(f"saved paired latent cache: {paired_cache}")
        else:
            model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
            direct_data = encode_paired_rows(model, h5_path, rows_by_split, args.encode_batch_size, list(args.perturbations), args)
            paired_metadata = paired_expected
            save_paired_cache(paired_cache, direct_data, paired_metadata)
            print(f"saved paired latent cache: {paired_cache}")

    rollout_data = None
    rollout_paired_metadata = None
    rollout_total = args.rollout_max_samples if args.rollout_max_samples is not None else args.max_samples
    rollout_counts = split_sample_counts(int(rollout_total), split_cfg, rows_by_split)
    rollout_expected = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "source_rows": {split: rows_fingerprint(rows) for split, rows in rows_by_split.items()},
        "num_samples": rollout_counts,
        "context_steps": args.context_steps,
        "frameskip": args.frameskip,
        "rollout_horizon": args.rollout_horizon,
        "normalize_actions": not args.no_normalize_actions,
        "perturbation_impl_version": PERTURBATION_IMPL_VERSION,
        "perturbations": list(args.perturbations),
        "agent_color": list(args.agent_color),
        "block_color": list(args.block_color),
        "target_color": list(args.target_color),
        "brightness_delta": args.brightness_delta,
        "noise_std": args.noise_std,
        "blur_radius": args.blur_radius,
        "mask_border_margin": args.mask_border_margin,
        "seed": args.seed,
    }
    if needs_rollout:
        if rollout_paired_cache.exists() and not args.force_recache:
            rollout_data, rollout_paired_metadata = load_paired_cache(rollout_paired_cache)
            if paired_cache_matches(rollout_paired_metadata, rollout_expected):
                print(f"loaded rollout paired latent cache: {rollout_paired_cache}")
            else:
                print(f"rollout paired cache metadata mismatch, recaching: {rollout_paired_cache}")
                model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
                rollout_data = encode_rollout_paired_rows(
                    model,
                    h5_path,
                    rows_by_split,
                    rollout_counts,
                    args.rollout_batch_size,
                    list(args.perturbations),
                    args,
                )
                rollout_paired_metadata = rollout_expected
                save_paired_cache(rollout_paired_cache, rollout_data, rollout_paired_metadata)
                print(f"saved rollout paired latent cache: {rollout_paired_cache}")
        else:
            model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
            rollout_data = encode_rollout_paired_rows(
                model,
                h5_path,
                rows_by_split,
                rollout_counts,
                args.rollout_batch_size,
                list(args.perturbations),
                args,
            )
            rollout_paired_metadata = rollout_expected
            save_paired_cache(rollout_paired_cache, rollout_data, rollout_paired_metadata)
            print(f"saved rollout paired latent cache: {rollout_paired_cache}")

    data = merge_pair_sources(direct_data, rollout_data, args.training_sources)
    results = train_all(data, args)
    results["clean_latent_cache"] = str(clean_cache)
    results["paired_latent_cache"] = str(paired_cache) if needs_direct else None
    results["rollout_paired_latent_cache"] = str(rollout_paired_cache) if needs_rollout else None
    results["clean_latent_cache_metadata"] = clean_metadata
    results["paired_latent_cache_metadata"] = paired_metadata
    results["rollout_paired_latent_cache_metadata"] = rollout_paired_metadata
    results_path = output_dir / "metrics.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"saved metrics: {results_path}")


if __name__ == "__main__":
    main()
