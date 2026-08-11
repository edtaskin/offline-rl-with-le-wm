"""Train a dense reward classifier on frozen PushT LeWM latents.

The classifier predicts whether the objective will be met within several future
world-model horizons. With the default horizons 2, 5, 10, 16 and frameskip 5,
the four sigmoid heads mean success within 10, 25, 50, and 80 environment
steps. Already-successful states are positive for every head.

Labels are derived from full expert trajectories using cached latent row IDs.
Rows too close to the end of an episode are masked for a head when the requested
lookahead is not fully observable and no success occurs before episode end.
"""

from __future__ import annotations

import argparse
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.train_state import (  # noqa: E402
    SplitConfig,
    dataset_path,
    encode_rows,
    load_latent_cache,
    repo_path,
    sample_rows,
    save_latent_cache,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class DenseRewardClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        monotonic_outputs: bool = False,
    ):
        super().__init__()
        self.monotonic_outputs = monotonic_outputs
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if not self.monotonic_outputs or logits.shape[-1] <= 1:
            return logits
        probs = torch.sigmoid(logits)
        probs = torch.cummax(probs, dim=-1).values
        eps = torch.finfo(probs.dtype).eps
        return torch.logit(probs.clamp(eps, 1.0 - eps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/probes/pusht_dense_reward")
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--label-cache", default=None)
    parser.add_argument("--imagined-cache", default=None)
    parser.add_argument("--imagined-val-cache", default=None)
    parser.add_argument("--imagined-test-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument("--sample-block-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 5, 10, 16])
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--imagined-rollout-horizon", type=int, default=None)
    parser.add_argument("--include-imagined-rollouts", action="store_true")
    parser.add_argument("--imagined-fraction", type=float, default=0.5)
    parser.add_argument("--eval-imagined-rollouts", action="store_true")
    parser.add_argument("--imagined-eval-samples", type=int, default=100_000)
    parser.add_argument(
        "--threshold-policy",
        choices=["f1", "target_fpr"],
        default="f1",
        help="Use validation F1 or highest-recall threshold with validation FPR <= target.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--monotonic-outputs", action="store_true")
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-mining", action="store_true")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--hard-negative-epochs", type=int, default=10)
    parser.add_argument("--hard-negative-threshold-source", choices=["thresholds", "target_fpr"], default="thresholds")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--force-label-recache", action="store_true")
    parser.add_argument("--force-imagined-recache", action="store_true")
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM rollouts.",
    )
    parser.add_argument(
        "--keep-censored-negatives",
        action="store_true",
        help="Treat near-episode-end unseen future failures as valid negatives instead of masking them.",
    )
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    return parser.parse_args()


def angle_delta(angle: np.ndarray, reference: float | np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle - reference), np.cos(angle - reference))


def objective_met_from_state(state: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    objective_pos = np.array([args.objective_x, args.objective_y], dtype=np.float32)
    pos_err = np.linalg.norm(state[:, 2:4] - objective_pos, axis=1)
    angle_err = np.abs(angle_delta(state[:, 4], float(args.objective_angle)))
    return (pos_err <= args.objective_pos_tol) & (angle_err <= args.objective_angle_tol)


def dense_labels_for_rows(
    rows: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    met_prefix: np.ndarray,
    horizons: list[int],
    frameskip: int,
    keep_censored_negatives: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = rows.astype(np.int64)
    episode = np.searchsorted(offsets, rows, side="right") - 1
    episode_end = offsets[episode] + lengths[episode] - 1
    y = np.zeros((len(rows), len(horizons)), dtype=np.float32)
    valid = np.zeros_like(y)

    for j, horizon in enumerate(horizons):
        lookahead = int(horizon) * int(frameskip)
        requested_end = rows + lookahead
        observed_end = np.minimum(requested_end, episode_end)
        any_success = (met_prefix[observed_end + 1] - met_prefix[rows]) > 0
        full_window_observed = requested_end <= episode_end
        y[:, j] = any_success.astype(np.float32)
        if keep_censored_negatives:
            valid[:, j] = 1.0
        else:
            valid[:, j] = np.logical_or(any_success, full_window_observed).astype(np.float32)
    return y, valid


def derive_dense_labels(
    h5_path: Path,
    rows_by_split: dict[str, np.ndarray],
    horizons: list[int],
    frameskip: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray]]:
    with h5py.File(h5_path, "r") as f:
        lengths = f["ep_len"][:].astype(np.int64)
        offsets = f["ep_offset"][:].astype(np.int64)
        state = f["state"][:].astype(np.float32)

    met_prefix = np.concatenate([[0], np.cumsum(objective_met_from_state(state, args).astype(np.int64))])
    output: dict[str, dict[str, np.ndarray]] = {}

    for split, rows in rows_by_split.items():
        rows = rows.astype(np.int64)
        y, valid = dense_labels_for_rows(
            rows,
            offsets,
            lengths,
            met_prefix,
            horizons,
            frameskip,
            args.keep_censored_negatives,
        )
        output[split] = {"y": y, "valid": valid, "rows": rows}
    return output


def save_label_cache(path: Path, labels: dict[str, dict[str, np.ndarray]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for split, split_data in labels.items():
        for key, value in split_data.items():
            arrays[f"{split}_{key}"] = value
    np.savez_compressed(path, **arrays)


def load_label_cache(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    raw = np.load(path, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    labels = {}
    for split in ("train", "val", "test"):
        labels[split] = {
            "y": raw[f"{split}_y"],
            "valid": raw[f"{split}_valid"],
            "rows": raw[f"{split}_rows"],
        }
    return labels, metadata


def label_metadata_matches(found: dict, expected: dict) -> bool:
    keys = (
        "dataset",
        "horizons",
        "frameskip",
        "objective_x",
        "objective_y",
        "objective_angle",
        "objective_pos_tol",
        "objective_angle_tol",
        "keep_censored_negatives",
        "latent_cache_rows",
    )
    return all(found.get(key) == expected.get(key) for key in keys)


def default_latent_cache(output_dir: Path, args: argparse.Namespace) -> Path:
    existing_1m = repo_path("models/probes/pusht_lewm_1M/latents.npz")
    if args.max_samples == 1_000_000 and existing_1m.exists():
        return existing_1m
    return output_dir / "latents.npz"


def preprocess_context_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 4, 2, 3).div_(255.0)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:]
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def sample_rollout_starts(
    offsets: np.ndarray,
    lengths: np.ndarray,
    train_rows: np.ndarray,
    num_windows: int,
    context_steps: int,
    horizon: int,
    frameskip: int,
    seed: int,
) -> list[tuple[int, int]]:
    train_episodes = np.unique(np.searchsorted(offsets, train_rows, side="right") - 1)
    required_last_frame = (context_steps + horizon - 1) * frameskip
    valid_eps = train_episodes[lengths[train_episodes] > required_last_frame]
    if len(valid_eps) == 0:
        raise ValueError("No train episodes are long enough for imagined rollout generation.")

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


def load_rollout_batch(
    h5: h5py.File,
    starts: list[tuple[int, int]],
    context_steps: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalize_actions: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bsz = len(starts)
    action_steps = context_steps + horizon - 1
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    action_blocks = np.empty((bsz, action_steps, frameskip * 2), dtype=np.float32)
    future_rows = np.empty((bsz, horizon), dtype=np.int64)
    episodes = np.empty((bsz,), dtype=np.int64)

    for i, (ep, start) in enumerate(starts):
        episodes[i] = ep
        context_idx = start + np.arange(context_steps) * frameskip
        context_pixels[i] = h5["pixels"][context_idx]
        future_rows[i] = start + (context_steps + np.arange(horizon)) * frameskip
        for t in range(action_steps):
            a0 = start + t * frameskip
            raw_action = h5["action"][a0 : a0 + frameskip].astype(np.float32)
            if normalize_actions:
                raw_action = (raw_action - action_mean) / action_std
            action_blocks[i, t] = raw_action.reshape(-1)
    return context_pixels, action_blocks, future_rows, episodes


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


def save_imagined_cache(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": np.array(json.dumps(metadata))}
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def load_imagined_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = np.load(path, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    arrays = {key: raw[key] for key in raw.files if key != "metadata"}
    return arrays, metadata


def imagined_metadata_matches(found: dict, expected: dict) -> bool:
    keys = (
        "dataset",
        "checkpoint",
        "horizons",
        "frameskip",
        "context_steps",
        "rollout_horizon",
        "num_samples",
        "split",
        "objective_x",
        "objective_y",
        "objective_angle",
        "objective_pos_tol",
        "objective_angle_tol",
        "keep_censored_negatives",
        "normalize_actions",
    )
    return all(found.get(key) == expected.get(key) for key in keys)


def build_imagined_train_cache(
    h5_path: Path,
    model: nn.Module,
    source_rows: np.ndarray,
    num_samples: int,
    rollout_horizon: int,
    args: argparse.Namespace,
    split: str,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    device = torch.device(args.device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))

    with h5py.File(h5_path, "r") as h5:
        offsets = h5["ep_offset"][:].astype(np.int64)
        lengths = h5["ep_len"][:].astype(np.int64)
        state = h5["state"][:].astype(np.float32)
        action_mean, action_std = action_stats(h5)
        starts = sample_rollout_starts(
            offsets,
            lengths,
            source_rows,
            num_windows=int(np.ceil(num_samples / rollout_horizon)),
            context_steps=args.context_steps,
            horizon=rollout_horizon,
            frameskip=args.frameskip,
            seed=args.seed + seed_offset,
        )
        met_prefix = np.concatenate([[0], np.cumsum(objective_met_from_state(state, args).astype(np.int64))])

        z_parts = []
        row_parts = []
        episode_parts = []
        model_step_parts = []
        start_parts = []
        y_parts = []
        valid_parts = []
        for start_idx in range(0, len(starts), args.rollout_batch_size):
            batch_starts = starts[start_idx : start_idx + args.rollout_batch_size]
            context_pixels, action_blocks, future_rows, episodes = load_rollout_batch(
                h5,
                batch_starts,
                context_steps=args.context_steps,
                horizon=rollout_horizon,
                frameskip=args.frameskip,
                action_mean=action_mean,
                action_std=action_std,
                normalize_actions=not args.no_normalize_actions,
            )
            emb = rollout_embeddings(model, context_pixels, action_blocks, rollout_horizon, history_size, device)
            flat_rows = future_rows.reshape(-1)
            y, valid = dense_labels_for_rows(
                flat_rows,
                offsets,
                lengths,
                met_prefix,
                args.horizons,
                args.frameskip,
                args.keep_censored_negatives,
            )
            z_parts.append(emb.reshape(-1, emb.shape[-1]).numpy())
            row_parts.append(flat_rows)
            episode_parts.append(np.repeat(episodes, rollout_horizon))
            model_step_parts.append(np.tile(np.arange(1, rollout_horizon + 1, dtype=np.int64), len(episodes)))
            start_parts.append(np.repeat(np.asarray([s for _, s in batch_starts], dtype=np.int64), rollout_horizon))
            y_parts.append(y)
            valid_parts.append(valid)
            done = min(sum(len(part) for part in row_parts), num_samples)
            print(f"imagined rollout cache: {done}/{num_samples}")

    z = np.concatenate(z_parts, axis=0)[:num_samples].astype(np.float32)
    rows = np.concatenate(row_parts, axis=0)[:num_samples].astype(np.int64)
    y = np.concatenate(y_parts, axis=0)[:num_samples].astype(np.float32)
    valid = np.concatenate(valid_parts, axis=0)[:num_samples].astype(np.float32)
    return {
        "z": z,
        "y": y,
        "valid": valid,
        "rows": rows,
        "episode": np.concatenate(episode_parts, axis=0)[:num_samples].astype(np.int64),
        "model_step": np.concatenate(model_step_parts, axis=0)[:num_samples].astype(np.int64),
        "start": np.concatenate(start_parts, axis=0)[:num_samples].astype(np.int64),
        "history_size": np.asarray([history_size], dtype=np.int64),
    }


def imagined_metadata_expected(
    args: argparse.Namespace,
    rollout_horizon: int,
    num_samples: int,
    split: str,
) -> dict:
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "horizons": args.horizons,
        "frameskip": args.frameskip,
        "context_steps": args.context_steps,
        "rollout_horizon": rollout_horizon,
        "num_samples": int(num_samples),
        "split": split,
        "objective_x": args.objective_x,
        "objective_y": args.objective_y,
        "objective_angle": args.objective_angle,
        "objective_pos_tol": args.objective_pos_tol,
        "objective_angle_tol": args.objective_angle_tol,
        "keep_censored_negatives": args.keep_censored_negatives,
        "normalize_actions": not args.no_normalize_actions,
    }


def get_or_build_imagined_cache(
    cache_path: Path,
    h5_path: Path,
    model: nn.Module,
    source_rows: np.ndarray,
    num_samples: int,
    rollout_horizon: int,
    args: argparse.Namespace,
    split: str,
    seed_offset: int,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    expected = imagined_metadata_expected(args, rollout_horizon, num_samples, split)
    if cache_path.exists() and not force:
        arrays, metadata = load_imagined_cache(cache_path)
        if imagined_metadata_matches(metadata, expected):
            print(f"loaded imagined {split} cache: {cache_path}")
            return arrays, metadata
        print(f"imagined {split} cache metadata mismatch, recaching: {cache_path}")

    arrays = build_imagined_train_cache(
        h5_path=h5_path,
        model=model,
        source_rows=source_rows,
        num_samples=num_samples,
        rollout_horizon=rollout_horizon,
        args=args,
        split=split,
        seed_offset=seed_offset,
    )
    metadata = expected
    save_imagined_cache(cache_path, arrays, metadata)
    print(f"saved imagined {split} cache: {cache_path}")
    return arrays, metadata


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def monotonic_consistency_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] <= 1:
        return logits.new_tensor(0.0)
    probs = torch.sigmoid(logits)
    violations = torch.relu(probs[..., :-1] - probs[..., 1:])
    return violations.mean()


def pos_weight_for(y: np.ndarray, valid: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = (y * valid).sum(axis=0)
    neg = ((1.0 - y) * valid).sum(axis=0)
    weight = np.divide(neg, np.maximum(pos, 1.0), out=np.ones_like(neg, dtype=np.float32), where=pos > 0)
    return torch.from_numpy(weight.astype(np.float32)).to(device)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    valid_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    valid_val: np.ndarray,
    args: argparse.Namespace,
    model: DenseRewardClassifier | None = None,
    epochs: int | None = None,
    stage: str = "train",
) -> DenseRewardClassifier:
    device = torch.device(args.device)
    if model is None:
        model = DenseRewardClassifier(
            x_train.shape[1],
            y_train.shape[1],
            args.hidden_dim,
            args.depth,
            monotonic_outputs=args.monotonic_outputs,
        )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = pos_weight_for(y_train, valid_train, device)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(valid_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    valid_val_t = torch.from_numpy(valid_val).float().to(device)

    best_loss = float("inf")
    best_state = None
    stale = 0
    max_epochs = int(epochs or args.epochs)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb, vb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)
            logits = model(xb)
            loss = masked_bce_loss(logits, yb, vb, pos_weight)
            if args.monotonic_loss_weight > 0:
                loss = loss + args.monotonic_loss_weight * monotonic_consistency_loss(logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val_t)
            val_loss_t = masked_bce_loss(val_logits, y_val_t, valid_val_t, pos_weight)
            if args.monotonic_loss_weight > 0:
                val_loss_t = val_loss_t + args.monotonic_loss_weight * monotonic_consistency_loss(val_logits)
            val_loss = val_loss_t.item()
        print(f"{stage} epoch {epoch:03d} val_loss={val_loss:.6f}")
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


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    probs = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).float()
        probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs, axis=0)


def binary_confusion(y_true: np.ndarray, y_prob: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    mask = valid.astype(bool)
    y = y_true[mask].astype(bool)
    pred = y_prob[mask] >= threshold
    tp = float(np.logical_and(pred, y).sum())
    tn = float(np.logical_and(~pred, ~y).sum())
    fp = float(np.logical_and(pred, ~y).sum())
    fn = float(np.logical_and(~pred, y).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray, valid: np.ndarray) -> float:
    mask = valid.astype(bool)
    y = y_true[mask].astype(np.int32)
    p = y_prob[mask]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray, valid: np.ndarray) -> float:
    mask = valid.astype(bool)
    y = y_true[mask].astype(np.int32)
    p = y_prob[mask]
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / pos)


def select_thresholds(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    valid_val: np.ndarray,
    policy: str,
    target_fpr: float,
) -> np.ndarray:
    thresholds = []
    for head in range(y_val.shape[1]):
        mask = valid_val[:, head].astype(bool)
        p = prob_val[mask, head]
        y = y_val[mask, head]
        candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
        candidates = np.concatenate([[0.5], candidates])
        best_threshold = 1.0
        best_score = -1.0
        for threshold in candidates:
            metrics = binary_confusion(y, p, np.ones_like(y), float(threshold))
            if policy == "f1":
                score = metrics["f1"]
            elif policy == "target_fpr":
                fp = metrics["fp"]
                tn = metrics["tn"]
                fpr = fp / max(fp + tn, 1.0)
                if fpr > target_fpr:
                    continue
                score = metrics["recall"]
            else:
                raise ValueError(f"Unknown threshold policy: {policy}")
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype=np.float32)


def evaluate_split(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    valid: np.ndarray,
    thresholds: np.ndarray,
    horizons: list[int],
    frameskip: int,
) -> dict:
    heads = {}
    for i, horizon in enumerate(horizons):
        mask = valid[:, i].astype(bool)
        y = y_true[:, i]
        p = y_prob[:, i]
        heads[str(horizon)] = {
            "world_model_steps": int(horizon),
            "env_steps": int(horizon * frameskip),
            "threshold": float(thresholds[i]),
            "valid_count": int(mask.sum()),
            "positive_count": int((y[mask] == 1).sum()),
            "negative_count": int((y[mask] == 0).sum()),
            "positive_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            "mean_probability": float(p[mask].mean()) if mask.any() else float("nan"),
            "roc_auc": roc_auc(y, p, valid[:, i]),
            "average_precision": average_precision(y, p, valid[:, i]),
            **binary_confusion(y, p, valid[:, i], float(thresholds[i])),
        }
    return {"heads": heads}


def hard_negative_indices(
    y: np.ndarray,
    prob: np.ndarray,
    valid: np.ndarray,
    thresholds: np.ndarray,
    max_count: int,
) -> np.ndarray:
    if max_count <= 0:
        return np.array([], dtype=np.int64)
    negative_valid = (valid > 0) & (y <= 0)
    excess = np.where(negative_valid, prob - thresholds[None, :], -np.inf)
    hard_score = excess.max(axis=1)
    candidates = np.nonzero(hard_score > 0)[0]
    if len(candidates) == 0:
        return candidates.astype(np.int64)
    order = np.argsort(-hard_score[candidates])
    return candidates[order[:max_count]].astype(np.int64)


def merge_latents_and_labels(
    data: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    merged = {}
    for split in ("train", "val", "test"):
        if not np.array_equal(data[split]["rows"], labels[split]["rows"]):
            raise ValueError(f"Latent and label rows differ for split {split}; recache labels.")
        merged[split] = {
            "z": data[split]["z"].astype(np.float32),
            "y": labels[split]["y"].astype(np.float32),
            "valid": labels[split]["valid"].astype(np.float32),
            "rows": data[split]["rows"],
        }
    return merged


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not 0.0 <= args.imagined_fraction <= 1.0:
        raise ValueError("--imagined-fraction must be between 0 and 1.")
    rollout_horizon = int(args.imagined_rollout_horizon or max(args.horizons))

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_path(args.cache_dir, args.dataset)
    cache_dir = repo_path(args.cache_dir)
    split_cfg = SplitConfig(seed=args.seed)

    latent_cache = repo_path(args.latent_cache) if args.latent_cache else default_latent_cache(output_dir, args)
    label_cache = repo_path(args.label_cache) if args.label_cache else output_dir / "dense_reward_labels.npz"
    imagined_cache = (
        repo_path(args.imagined_cache) if args.imagined_cache else output_dir / "imagined_train_rollouts.npz"
    )
    imagined_val_cache = (
        repo_path(args.imagined_val_cache) if args.imagined_val_cache else output_dir / "imagined_val_rollouts.npz"
    )
    imagined_test_cache = (
        repo_path(args.imagined_test_cache) if args.imagined_test_cache else output_dir / "imagined_test_rollouts.npz"
    )

    if latent_cache.exists() and not args.force_recache:
        data, latent_metadata = load_latent_cache(latent_cache)
        print(f"loaded latent cache: {latent_cache}")
    else:
        rows = sample_rows(h5_path, args.max_samples, args.sample_block_size, split_cfg)
        model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        data = encode_rows(
            model=model,
            h5_path=h5_path,
            rows_by_split=rows,
            batch_size=args.encode_batch_size,
            device=torch.device(args.device),
        )
        latent_metadata = {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "cache_dir": str(cache_dir),
            "max_samples": args.max_samples,
            "sample_block_size": args.sample_block_size,
            "split": asdict(split_cfg),
        }
        save_latent_cache(latent_cache, data, latent_metadata)
        print(f"saved latent cache: {latent_cache}")

    label_metadata_expected = {
        "dataset": args.dataset,
        "horizons": args.horizons,
        "frameskip": args.frameskip,
        "objective_x": args.objective_x,
        "objective_y": args.objective_y,
        "objective_angle": args.objective_angle,
        "objective_pos_tol": args.objective_pos_tol,
        "objective_angle_tol": args.objective_angle_tol,
        "keep_censored_negatives": args.keep_censored_negatives,
        "latent_cache_rows": {split: int(len(data[split]["rows"])) for split in data},
    }
    if label_cache.exists() and not args.force_label_recache:
        labels, label_metadata = load_label_cache(label_cache)
        if label_metadata_matches(label_metadata, label_metadata_expected):
            print(f"loaded label cache: {label_cache}")
        else:
            print(f"label cache metadata mismatch, recaching: {label_cache}")
            labels = derive_dense_labels(
                h5_path=h5_path,
                rows_by_split={split: split_data["rows"] for split, split_data in data.items()},
                horizons=args.horizons,
                frameskip=args.frameskip,
                args=args,
            )
            label_metadata = label_metadata_expected
            save_label_cache(label_cache, labels, label_metadata)
            print(f"saved label cache: {label_cache}")
    else:
        labels = derive_dense_labels(
            h5_path=h5_path,
            rows_by_split={split: split_data["rows"] for split, split_data in data.items()},
            horizons=args.horizons,
            frameskip=args.frameskip,
            args=args,
        )
        label_metadata = label_metadata_expected
        save_label_cache(label_cache, labels, label_metadata)
        print(f"saved label cache: {label_cache}")

    merged = merge_latents_and_labels(data, labels)
    train_source_counts = {"gt": int(len(merged["train"]["z"])), "imagined": 0}
    imagined_metadata = None
    imagined_val = None
    imagined_test = None
    imagined_val_metadata = None
    imagined_test_metadata = None
    rollout_model = None

    train_z_raw = merged["train"]["z"]
    y_train = merged["train"]["y"]
    valid_train = merged["train"]["valid"]
    train_rows = merged["train"]["rows"]

    if args.include_imagined_rollouts and args.imagined_fraction > 0:
        n_train_total = len(train_z_raw)
        n_imagined = int(round(n_train_total * args.imagined_fraction))
        n_gt_keep = n_train_total - n_imagined
        rollout_model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        imagined, imagined_metadata = get_or_build_imagined_cache(
            cache_path=imagined_cache,
            h5_path=h5_path,
            model=rollout_model,
            source_rows=train_rows,
            num_samples=n_imagined,
            rollout_horizon=rollout_horizon,
            args=args,
            split="train",
            seed_offset=17,
            force=args.force_imagined_recache,
        )

        rng = np.random.default_rng(args.seed + 23)
        gt_keep_idx = rng.choice(n_train_total, size=n_gt_keep, replace=False) if n_gt_keep > 0 else np.array([], dtype=np.int64)
        train_z_raw = np.concatenate([train_z_raw[gt_keep_idx], imagined["z"].astype(np.float32)], axis=0)
        y_train = np.concatenate([y_train[gt_keep_idx], imagined["y"].astype(np.float32)], axis=0)
        valid_train = np.concatenate([valid_train[gt_keep_idx], imagined["valid"].astype(np.float32)], axis=0)
        train_source_counts = {"gt": int(n_gt_keep), "imagined": int(n_imagined)}
        print(f"training mix: {n_gt_keep} GT latents + {n_imagined} imagined rollout latents")

    if args.eval_imagined_rollouts:
        if rollout_model is None:
            rollout_model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        imagined_val, imagined_val_metadata = get_or_build_imagined_cache(
            cache_path=imagined_val_cache,
            h5_path=h5_path,
            model=rollout_model,
            source_rows=merged["val"]["rows"],
            num_samples=args.imagined_eval_samples,
            rollout_horizon=rollout_horizon,
            args=args,
            split="val",
            seed_offset=31,
            force=args.force_imagined_recache,
        )
        imagined_test, imagined_test_metadata = get_or_build_imagined_cache(
            cache_path=imagined_test_cache,
            h5_path=h5_path,
            model=rollout_model,
            source_rows=merged["test"]["rows"],
            num_samples=args.imagined_eval_samples,
            rollout_horizon=rollout_horizon,
            args=args,
            split="test",
            seed_offset=47,
            force=args.force_imagined_recache,
        )

    x_train, (x_val, x_test), x_mean, x_std = standardize(
        train_z_raw, merged["val"]["z"], merged["test"]["z"]
    )
    y_val = merged["val"]["y"]
    y_test = merged["test"]["y"]
    valid_val = merged["val"]["valid"]
    valid_test = merged["test"]["valid"]

    print("valid/positive counts by horizon:")
    for i, horizon in enumerate(args.horizons):
        print(
            f"  {horizon} wm ({horizon * args.frameskip} env): "
            f"train valid={int(valid_train[:, i].sum())} pos={int((y_train[:, i] * valid_train[:, i]).sum())}, "
            f"val valid={int(valid_val[:, i].sum())} pos={int((y_val[:, i] * valid_val[:, i]).sum())}, "
            f"test valid={int(valid_test[:, i].sum())} pos={int((y_test[:, i] * valid_test[:, i]).sum())}"
        )

    classifier = train_model(x_train, y_train, valid_train, x_val, y_val, valid_val, args, stage="train")
    val_prob = predict(classifier, x_val, args.batch_size)
    thresholds = select_thresholds(y_val, val_prob, valid_val, args.threshold_policy, args.target_fpr)
    hard_negative_count = 0
    if args.hard_negative_mining:
        train_prob = predict(classifier, x_train, args.batch_size)
        mining_policy = "target_fpr" if args.hard_negative_threshold_source == "target_fpr" else args.threshold_policy
        mining_thresholds = (
            select_thresholds(y_val, val_prob, valid_val, "target_fpr", args.target_fpr)
            if mining_policy == "target_fpr"
            else thresholds
        )
        max_hard = int(round(len(x_train) * args.hard_negative_fraction))
        hard_idx = hard_negative_indices(y_train, train_prob, valid_train, mining_thresholds, max_hard)
        hard_negative_count = int(len(hard_idx))
        if len(hard_idx) > 0:
            print(f"hard-negative mining: oversampling {len(hard_idx)} high-scoring valid negatives")
            x_train_hn = np.concatenate([x_train, x_train[hard_idx]], axis=0)
            y_train_hn = np.concatenate([y_train, y_train[hard_idx]], axis=0)
            valid_train_hn = np.concatenate([valid_train, valid_train[hard_idx]], axis=0)
            classifier = train_model(
                x_train_hn,
                y_train_hn,
                valid_train_hn,
                x_val,
                y_val,
                valid_val,
                args,
                model=classifier,
                epochs=args.hard_negative_epochs,
                stage="hard-negative",
            )
            val_prob = predict(classifier, x_val, args.batch_size)
            thresholds = select_thresholds(y_val, val_prob, valid_val, args.threshold_policy, args.target_fpr)
        else:
            print("hard-negative mining: no high-scoring valid negatives found")

    test_prob = predict(classifier, x_test, args.batch_size)
    val_metrics = evaluate_split(y_val, val_prob, valid_val, thresholds, args.horizons, args.frameskip)
    test_metrics = evaluate_split(y_test, test_prob, valid_test, thresholds, args.horizons, args.frameskip)
    imagined_val_metrics = None
    imagined_test_metrics = None
    if imagined_val is not None and imagined_test is not None:
        imagined_val_x = apply_standardizer(imagined_val["z"], x_mean, x_std)
        imagined_test_x = apply_standardizer(imagined_test["z"], x_mean, x_std)
        imagined_val_prob = predict(classifier, imagined_val_x, args.batch_size)
        imagined_test_prob = predict(classifier, imagined_test_x, args.batch_size)
        imagined_val_metrics = evaluate_split(
            imagined_val["y"],
            imagined_val_prob,
            imagined_val["valid"],
            thresholds,
            args.horizons,
            args.frameskip,
        )
        imagined_test_metrics = evaluate_split(
            imagined_test["y"],
            imagined_test_prob,
            imagined_test["valid"],
            thresholds,
            args.horizons,
            args.frameskip,
        )

    torch.save(
        {
            "model": classifier.state_dict(),
            "input_dim": x_train.shape[1],
            "output_dim": len(args.horizons),
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "monotonic_outputs": args.monotonic_outputs,
            "monotonic_loss_weight": args.monotonic_loss_weight,
            "hard_negative_mining": args.hard_negative_mining,
            "hard_negative_fraction": args.hard_negative_fraction,
            "hard_negative_epochs": args.hard_negative_epochs,
            "hard_negative_count": hard_negative_count,
            "horizons": args.horizons,
            "frameskip": args.frameskip,
            "context_steps": args.context_steps,
            "include_imagined_rollouts": args.include_imagined_rollouts,
            "imagined_fraction": args.imagined_fraction,
            "imagined_rollout_horizon": rollout_horizon,
            "thresholds": thresholds,
            "threshold_policy": args.threshold_policy,
            "target_fpr": args.target_fpr,
            "x_mean": x_mean,
            "x_std": x_std,
            "objective": {
                "x": args.objective_x,
                "y": args.objective_y,
                "angle": args.objective_angle,
                "pos_tol": args.objective_pos_tol,
                "angle_tol": args.objective_angle_tol,
            },
            "task": "dense_time_to_success_binary",
        },
        output_dir / "dense_reward_classifier.pt",
    )

    summary = {
        "config": vars(args),
        "latent_cache": str(latent_cache),
        "label_cache": str(label_cache),
        "imagined_cache": str(imagined_cache) if args.include_imagined_rollouts else None,
        "imagined_val_cache": str(imagined_val_cache) if args.eval_imagined_rollouts else None,
        "imagined_test_cache": str(imagined_test_cache) if args.eval_imagined_rollouts else None,
        "latent_cache_metadata": latent_metadata,
        "label_cache_metadata": label_metadata,
        "imagined_cache_metadata": imagined_metadata,
        "imagined_val_cache_metadata": imagined_val_metadata,
        "imagined_test_cache_metadata": imagined_test_metadata,
        "train_source_counts": train_source_counts,
        "normalizer": {"mean": x_mean.tolist(), "std": x_std.tolist()},
        "thresholds": thresholds.tolist(),
        "threshold_policy": args.threshold_policy,
        "target_fpr": args.target_fpr,
        "hard_negative_count": hard_negative_count,
        "val": val_metrics,
        "test": test_metrics,
        "imagined_val": imagined_val_metrics,
        "imagined_test": imagined_test_metrics,
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"thresholds": thresholds.tolist(), "test": test_metrics}, indent=2))
    print(f"saved classifier to {output_dir / 'dense_reward_classifier.pt'}")


if __name__ == "__main__":
    main()
