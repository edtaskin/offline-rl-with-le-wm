"""Train linear and MLP probes on frozen PushT LeWM latents.

The probes decode physical state variables from the projected CLS latent used
by LeWM for planning:
  - agent position: state[0:2]
  - block position: state[2:4]
  - block angle:    state[4], represented as sin/cos during training
  - block pose relative to objective
  - block pose relative to agent
  - objective met classification
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path

FEATURES = {
    "agent_pos": (0, 2),
    "block_pos": (2, 4),
    "block_angle": (4, 5),
    "block_rel_objective": None,
    "block_rel_agent": None,
    "objective_met": None,
}

CLASSIFICATION_FEATURES = {"objective_met"}
ANGLE_FEATURES = {"block_angle"}
RELATIVE_POSE_FEATURES = {"block_rel_objective", "block_rel_agent"}


@dataclass
class SplitConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    seed: int = 3072


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/probes/pusht_lewm")
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--sample-block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["all"],
        choices=["all", *FEATURES.keys()],
        help="Probe names to train. Use 'all' to train every probe.",
    )
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument(
        "--objective-pos-tol",
        type=float,
        default=20.0,
        help="Positive class threshold for objective_met, in pixels.",
    )
    parser.add_argument(
        "--objective-angle-tol",
        type=float,
        default=float(np.pi / 9),
        help="Positive class threshold for objective_met, in radians.",
    )
    return parser.parse_args()


def dataset_path(cache_dir: str | Path, dataset: str) -> Path:
    return repo_path(cache_dir) / "datasets" / f"{dataset}.h5"


def split_episodes(num_episodes: int, cfg: SplitConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    episodes = rng.permutation(num_episodes)
    n_train = int(num_episodes * cfg.train_fraction)
    n_val = int(num_episodes * cfg.val_fraction)
    return {
        "train": np.sort(episodes[:n_train]),
        "val": np.sort(episodes[n_train : n_train + n_val]),
        "test": np.sort(episodes[n_train + n_val :]),
    }


def rows_for_episodes(
    offsets: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
    max_rows: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_rows < 0:
        rows = np.concatenate(
            [np.arange(offsets[ep], offsets[ep] + lengths[ep], dtype=np.int64) for ep in episodes]
        )
        return np.sort(rows)

    block_size = max(1, block_size)
    rows = []
    seen = set()
    episode_lengths = lengths[episodes].astype(np.float64)
    episode_probs = episode_lengths / episode_lengths.sum()
    max_attempts = max(10_000, 20 * int(np.ceil(max_rows / block_size)))

    attempts = 0
    while len(rows) < max_rows and attempts < max_attempts:
        attempts += 1
        ep = int(rng.choice(episodes, p=episode_probs))
        ep_len = int(lengths[ep])
        start = int(rng.integers(0, max(1, ep_len - block_size + 1)))
        global_start = int(offsets[ep] + start)
        for row in range(global_start, min(global_start + block_size, int(offsets[ep] + ep_len))):
            if row not in seen:
                seen.add(row)
                rows.append(row)
                if len(rows) >= max_rows:
                    break

    if len(rows) < max_rows:
        fallback = np.concatenate(
            [np.arange(offsets[ep], offsets[ep] + lengths[ep], dtype=np.int64) for ep in episodes]
        )
        needed = min(max_rows - len(rows), len(fallback))
        candidates = np.setdiff1d(fallback, np.fromiter(seen, dtype=np.int64), assume_unique=False)
        if needed > 0 and len(candidates) > 0:
            rows.extend(rng.choice(candidates, size=min(needed, len(candidates)), replace=False).tolist())

    return np.sort(np.asarray(rows, dtype=np.int64))


def sample_rows(
    h5_path: Path,
    max_samples: int,
    block_size: int,
    split_cfg: SplitConfig,
) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        lengths = f["ep_len"][:]
        offsets = f["ep_offset"][:]

    splits = split_episodes(len(lengths), split_cfg)
    if max_samples < 0:
        per_split = {name: -1 for name in splits}
    else:
        per_split = {
            "train": int(max_samples * split_cfg.train_fraction),
            "val": int(max_samples * split_cfg.val_fraction),
            "test": max_samples
            - int(max_samples * split_cfg.train_fraction)
            - int(max_samples * split_cfg.val_fraction),
        }

    rng = np.random.default_rng(split_cfg.seed + 1)
    return {
        name: rows_for_episodes(offsets, lengths, episodes, per_split[name], block_size, rng)
        for name, episodes in splits.items()
    }


def preprocess_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2).div_(255.0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (x - mean) / std


def contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    breaks = np.nonzero(np.diff(rows) != 1)[0] + 1
    parts = np.split(rows, breaks)
    return [(int(part[0]), int(part[-1]) + 1) for part in parts]


def iter_h5_batches(
    h5_file: h5py.File,
    rows: np.ndarray,
    batch_size: int,
):
    pixels_parts = []
    state_parts = []
    n_buffered = 0

    for run_start, run_end in contiguous_runs(rows):
        for start in range(run_start, run_end, batch_size):
            end = min(start + batch_size, run_end)
            pixels_parts.append(h5_file["pixels"][start:end])
            state_parts.append(h5_file["state"][start:end].astype(np.float32))
            n_buffered += end - start

            if n_buffered >= batch_size:
                pixels = np.concatenate(pixels_parts, axis=0)
                states = np.concatenate(state_parts, axis=0)
                yield pixels[:batch_size], states[:batch_size]

                pixels_tail = pixels[batch_size:]
                states_tail = states[batch_size:]
                pixels_parts = [pixels_tail] if len(pixels_tail) else []
                state_parts = [states_tail] if len(states_tail) else []
                n_buffered = len(pixels_tail)

    if n_buffered:
        yield np.concatenate(pixels_parts, axis=0), np.concatenate(state_parts, axis=0)


@torch.inference_mode()
def encode_rows(
    model: nn.Module,
    h5_path: Path,
    rows_by_split: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    model.eval().to(device)
    output: dict[str, dict[str, np.ndarray]] = {}

    with h5py.File(h5_path, "r") as f:
        for split, rows in rows_by_split.items():
            latents = []
            labels = []
            for batch_idx, (pixel_batch, state_batch) in enumerate(iter_h5_batches(f, rows, batch_size)):
                pixels = preprocess_pixels(pixel_batch, device)
                info = {"pixels": pixels.unsqueeze(1)}
                emb = model.encode(info)["emb"][:, 0].detach().cpu().float().numpy()
                latents.append(emb)
                labels.append(state_batch)
                if (batch_idx + 1) % 25 == 0:
                    done = min((batch_idx + 1) * batch_size, len(rows))
                    print(f"encoding {split}: {done}/{len(rows)}")

            output[split] = {
                "z": np.concatenate(latents, axis=0),
                "state": np.concatenate(labels, axis=0),
                "rows": rows,
            }
            print(f"cached {split}: {len(rows)} samples")
    return output


def save_latent_cache(path: Path, data: dict[str, dict[str, np.ndarray]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for split, split_data in data.items():
        for key, value in split_data.items():
            arrays[f"{split}_{key}"] = value
    np.savez_compressed(path, **arrays)


def load_latent_cache(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    raw = np.load(path, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    data = {}
    for split in ("train", "val", "test"):
        data[split] = {
            "z": raw[f"{split}_z"],
            "state": raw[f"{split}_state"],
            "rows": raw[f"{split}_rows"],
        }
    return data, metadata


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def angle_delta(angle: np.ndarray, reference: float | np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle - reference), np.cos(angle - reference))


def rotate_2d(vec: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """Rotate 2D vectors by angle, broadcasting over rows."""
    c = np.cos(angle)
    s = np.sin(angle)
    x = vec[..., 0]
    y = vec[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def targets(state: np.ndarray, feature: str, args: argparse.Namespace) -> np.ndarray:
    objective_pos = np.array([args.objective_x, args.objective_y], dtype=np.float32)
    objective_angle = float(args.objective_angle)

    if feature == "block_angle":
        angle = state[:, 4]
        return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)
    if feature == "block_rel_objective":
        # Translation is block center relative to target center, expressed in
        # the objective frame. Angle is circular block/objective error.
        rel_xy = rotate_2d(state[:, 2:4] - objective_pos, -objective_angle)
        rel_angle = angle_delta(state[:, 4], objective_angle)
        return np.concatenate(
            [rel_xy, np.sin(rel_angle)[:, None], np.cos(rel_angle)[:, None]],
            axis=1,
        ).astype(np.float32)
    if feature == "block_rel_agent":
        # Translation is agent position relative to the block center, expressed
        # in the block frame. This captures how the pusher is placed around the T.
        rel_xy = rotate_2d(state[:, 0:2] - state[:, 2:4], -state[:, 4])
        return np.concatenate(
            [rel_xy, np.sin(state[:, 4])[:, None], np.cos(state[:, 4])[:, None]],
            axis=1,
        ).astype(np.float32)
    if feature == "objective_met":
        pos_err = np.linalg.norm(state[:, 2:4] - objective_pos, axis=1)
        angle_err = np.abs(angle_delta(state[:, 4], objective_angle))
        return ((pos_err <= args.objective_pos_tol) & (angle_err <= args.objective_angle_tol))[
            :, None
        ].astype(np.float32)
    start, end = FEATURES[feature]
    return state[:, start:end].astype(np.float32)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    penalty = ridge * np.eye(xtx.shape[0], dtype=x.dtype)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty, x_aug.T @ y)


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


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


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    output_dim: int,
) -> ProbeMLP:
    device = torch.device(args.device)
    model = ProbeMLP(x_train.shape[1], output_dim, args.mlp_hidden, args.mlp_depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    kind: str,
) -> tuple[nn.Module, float]:
    device = torch.device(args.device)
    if kind == "linear":
        model: nn.Module = LinearClassifier(x_train.shape[1], y_train.shape[1]).to(device)
    elif kind == "mlp":
        model = ProbeMLP(x_train.shape[1], y_train.shape[1], args.mlp_hidden, args.mlp_depth).to(device)
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if pos <= 0:
        raise ValueError("objective_met has no positive examples in the training split")
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    val_prob = predict_classifier(model, x_val, args.batch_size)
    threshold = select_threshold(y_val, val_prob)
    return model, threshold


@torch.inference_mode()
def predict_mlp(model: ProbeMLP, x: np.ndarray, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).float()
        preds.append(model(xb).numpy())
    return np.concatenate(preds, axis=0)


@torch.inference_mode()
def predict_classifier(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    probs = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).float()
        probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs, axis=0)


def pearsonr(pred: np.ndarray, true: np.ndarray) -> float:
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    true_centered = true - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred_centered**2).sum(axis=0) * (true_centered**2).sum(axis=0))
    corr = np.divide(
        (pred_centered * true_centered).sum(axis=0),
        denom,
        out=np.zeros(pred.shape[1], dtype=np.float64),
        where=denom > 1e-12,
    )
    return float(np.mean(corr))


def binary_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.reshape(-1).astype(bool)
    pred = y_prob.reshape(-1) >= threshold
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


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.reshape(-1).astype(np.int32)
    p = y_prob.reshape(-1)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.reshape(-1).astype(np.int32)
    p = y_prob.reshape(-1)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / pos)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "pearson": pearsonr(pred, true),
    }


def relative_pose_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    xy = regression_metrics(pred[:, :2], true[:, :2])
    angle = angle_metrics(pred[:, 2:4], true[:, 2:4])
    return {
        "xy_mse": xy["mse"],
        "xy_rmse": xy["rmse"],
        "xy_mae": xy["mae"],
        "xy_r2": xy["r2"],
        "xy_pearson": xy["pearson"],
        **angle,
    }


def classification_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    conf = binary_confusion(y_true, y_prob, threshold)
    prevalence = float(y_true.mean())
    return {
        "threshold": float(threshold),
        "positive_rate": prevalence,
        "mean_predicted_probability": float(y_prob.mean()),
        "roc_auc": roc_auc(y_true, y_prob),
        "average_precision": average_precision(y_true, y_prob),
        **conf,
    }


def angle_metrics(pred_sincos: np.ndarray, true_sincos: np.ndarray) -> dict[str, float]:
    pred_norm = pred_sincos / np.maximum(np.linalg.norm(pred_sincos, axis=1, keepdims=True), 1e-8)
    true_norm = true_sincos / np.maximum(np.linalg.norm(true_sincos, axis=1, keepdims=True), 1e-8)
    pred_angle = np.arctan2(pred_norm[:, 0], pred_norm[:, 1])
    true_angle = np.arctan2(true_norm[:, 0], true_norm[:, 1])
    diff = np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle))
    return {
        "sincos_mse": float(np.mean((pred_sincos - true_sincos) ** 2)),
        "circular_mae_rad": float(np.mean(np.abs(diff))),
        "circular_rmse_rad": float(np.sqrt(np.mean(diff**2))),
        "circular_mae_deg": float(np.degrees(np.mean(np.abs(diff)))),
        "mean_cosine": float(np.mean(np.sum(pred_norm * true_norm, axis=1))),
        "pearson_sincos": pearsonr(pred_sincos, true_sincos),
    }


def evaluate(feature: str, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    if feature == "block_angle":
        return angle_metrics(pred, true)
    if feature in RELATIVE_POSE_FEATURES:
        return relative_pose_metrics(pred, true)
    return regression_metrics(pred, true)


def selected_probes(args: argparse.Namespace) -> list[str]:
    if "all" in args.probes:
        return list(FEATURES.keys())
    return list(dict.fromkeys(args.probes))


def train_all_probes(data: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> dict:
    z_train, (z_val, z_test), z_mean, z_std = standardize(
        data["train"]["z"].astype(np.float32),
        data["val"]["z"].astype(np.float32),
        data["test"]["z"].astype(np.float32),
    )

    results = {
        "config": vars(args),
        "feature_normalizer": {"mean": z_mean.tolist(), "std": z_std.tolist()},
        "probes": {},
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.json"
    if results_path.exists() and "all" not in args.probes:
        with results_path.open() as f:
            previous = json.load(f)
        results["probes"] = previous.get("probes", {})
        results["previous_config"] = previous.get("config", {})

    for feature in selected_probes(args):
        y_train_raw = targets(data["train"]["state"], feature, args)
        y_val_raw = targets(data["val"]["state"], feature, args)
        y_test_raw = targets(data["test"]["state"], feature, args)

        feature_dir = output_dir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)

        if feature in CLASSIFICATION_FEATURES:
            y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw
            class_counts = {
                "train_positive": int(y_train.sum()),
                "train_negative": int(len(y_train) - y_train.sum()),
                "val_positive": int(y_val.sum()),
                "val_negative": int(len(y_val) - y_val.sum()),
                "test_positive": int(y_test.sum()),
                "test_negative": int(len(y_test) - y_test.sum()),
            }

            linear_clf, linear_threshold = train_classifier(
                z_train, y_train, z_val, y_val, args, kind="linear"
            )
            linear_prob = predict_classifier(linear_clf, z_test, args.batch_size)
            linear_metrics = classification_metrics(linear_prob, y_test, linear_threshold)

            mlp_clf, mlp_threshold = train_classifier(z_train, y_train, z_val, y_val, args, kind="mlp")
            mlp_prob = predict_classifier(mlp_clf, z_test, args.batch_size)
            mlp_metrics = classification_metrics(mlp_prob, y_test, mlp_threshold)

            torch.save(
                {
                    "model": linear_clf.state_dict(),
                    "input_dim": z_train.shape[1],
                    "output_dim": y_train.shape[1],
                    "x_mean": z_mean,
                    "x_std": z_std,
                    "threshold": linear_threshold,
                    "task": "binary_classification",
                },
                feature_dir / "linear_probe.pt",
            )
            torch.save(
                {
                    "model": mlp_clf.state_dict(),
                    "input_dim": z_train.shape[1],
                    "output_dim": y_train.shape[1],
                    "hidden_dim": args.mlp_hidden,
                    "depth": args.mlp_depth,
                    "x_mean": z_mean,
                    "x_std": z_std,
                    "threshold": mlp_threshold,
                    "task": "binary_classification",
                },
                feature_dir / "mlp_probe.pt",
            )

            results["probes"][feature] = {
                "class_counts": class_counts,
                "linear": linear_metrics,
                "mlp": mlp_metrics,
            }
            print(f"{feature} class_counts: {class_counts}")
            print(f"{feature} linear: {linear_metrics}")
            print(f"{feature} mlp:    {mlp_metrics}")
            continue

        if feature in ANGLE_FEATURES:
            y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw
            y_mean = np.zeros((1, y_train.shape[1]), dtype=np.float32)
            y_std = np.ones((1, y_train.shape[1]), dtype=np.float32)
        else:
            y_train, (y_val, y_test), y_mean, y_std = standardize(y_train_raw, y_val_raw, y_test_raw)

        ridge_w = fit_ridge(z_train, y_train, args.ridge)
        ridge_pred = predict_ridge(z_test, ridge_w) * y_std + y_mean
        linear_metrics = evaluate(feature, ridge_pred, y_test_raw)

        mlp = train_mlp(z_train, y_train, z_val, y_val, args, y_train.shape[1])
        mlp_pred = predict_mlp(mlp, z_test, args.batch_size) * y_std + y_mean
        mlp_metrics = evaluate(feature, mlp_pred, y_test_raw)

        np.savez(
            feature_dir / "linear_probe.npz",
            weights=ridge_w,
            x_mean=z_mean,
            x_std=z_std,
            y_mean=y_mean,
            y_std=y_std,
        )
        torch.save(
            {
                "model": mlp.state_dict(),
                "input_dim": z_train.shape[1],
                "output_dim": y_train.shape[1],
                "hidden_dim": args.mlp_hidden,
                "depth": args.mlp_depth,
                "x_mean": z_mean,
                "x_std": z_std,
                "y_mean": y_mean,
                "y_std": y_std,
            },
            feature_dir / "mlp_probe.pt",
        )

        results["probes"][feature] = {
            "linear": linear_metrics,
            "mlp": mlp_metrics,
        }
        print(f"{feature} linear: {linear_metrics}")
        print(f"{feature} mlp:    {mlp_metrics}")

    return results


def main() -> None:
    args = parse_args()
    h5_path = dataset_path(args.cache_dir, args.dataset)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    split_cfg = SplitConfig(seed=3072)
    output_dir = repo_path(args.output_dir)
    cache_dir = repo_path(args.cache_dir)
    latent_cache = repo_path(args.latent_cache) if args.latent_cache else output_dir / "latents.npz"

    if latent_cache.exists() and not args.force_recache:
        data, metadata = load_latent_cache(latent_cache)
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
        metadata = {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "cache_dir": str(cache_dir),
            "max_samples": args.max_samples,
            "sample_block_size": args.sample_block_size,
            "split": asdict(split_cfg),
        }
        save_latent_cache(latent_cache, data, metadata)
        print(f"saved latent cache: {latent_cache}")

    results = train_all_probes(data, args)
    results["latent_cache_metadata"] = metadata
    results_path = output_dir / "metrics.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"saved metrics: {results_path}")


if __name__ == "__main__":
    main()
