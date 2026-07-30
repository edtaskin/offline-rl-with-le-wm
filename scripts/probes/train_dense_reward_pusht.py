"""Train a dense reward classifier on frozen PushT LeWM latents.

The classifier predicts whether the objective will be met within several future
world-model horizons. With the default horizons 2, 5, 10, 20 and frameskip 5,
the four sigmoid heads mean success within 10, 25, 50, and 100 environment
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

from scripts.probes.train_probes_pusht import (  # noqa: E402
    SplitConfig,
    dataset_path,
    encode_rows,
    load_latent_cache,
    repo_path,
    sample_rows,
    save_latent_cache,
)


class DenseRewardClassifier(nn.Module):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/probes/pusht_dense_reward")
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--label-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument("--sample-block-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--force-label-recache", action="store_true")
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

    met = objective_met_from_state(state, args).astype(np.int64)
    met_prefix = np.concatenate([[0], np.cumsum(met)])
    output: dict[str, dict[str, np.ndarray]] = {}

    for split, rows in rows_by_split.items():
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
            if args.keep_censored_negatives:
                valid[:, j] = 1.0
            else:
                valid[:, j] = np.logical_or(any_success, full_window_observed).astype(np.float32)

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


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


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
) -> DenseRewardClassifier:
    device = torch.device(args.device)
    model = DenseRewardClassifier(x_train.shape[1], y_train.shape[1], args.hidden_dim, args.depth).to(device)
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, vb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)
            loss = masked_bce_loss(model(xb), yb, vb, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            val_loss = masked_bce_loss(model(x_val_t), y_val_t, valid_val_t, pos_weight).item()
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


def select_thresholds(y_val: np.ndarray, prob_val: np.ndarray, valid_val: np.ndarray) -> np.ndarray:
    thresholds = []
    for head in range(y_val.shape[1]):
        mask = valid_val[:, head].astype(bool)
        p = prob_val[mask, head]
        y = y_val[mask, head]
        candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
        candidates = np.concatenate([[0.5], candidates])
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            f1 = binary_confusion(y, p, np.ones_like(y), float(threshold))["f1"]
            if f1 > best_f1:
                best_f1 = f1
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

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_path(args.cache_dir, args.dataset)
    cache_dir = repo_path(args.cache_dir)
    split_cfg = SplitConfig(seed=args.seed)

    latent_cache = repo_path(args.latent_cache) if args.latent_cache else default_latent_cache(output_dir, args)
    label_cache = repo_path(args.label_cache) if args.label_cache else output_dir / "dense_reward_labels.npz"

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
    x_train, (x_val, x_test), x_mean, x_std = standardize(
        merged["train"]["z"], merged["val"]["z"], merged["test"]["z"]
    )
    y_train = merged["train"]["y"]
    y_val = merged["val"]["y"]
    y_test = merged["test"]["y"]
    valid_train = merged["train"]["valid"]
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

    classifier = train_model(x_train, y_train, valid_train, x_val, y_val, valid_val, args)
    val_prob = predict(classifier, x_val, args.batch_size)
    test_prob = predict(classifier, x_test, args.batch_size)
    thresholds = select_thresholds(y_val, val_prob, valid_val)
    val_metrics = evaluate_split(y_val, val_prob, valid_val, thresholds, args.horizons, args.frameskip)
    test_metrics = evaluate_split(y_test, test_prob, valid_test, thresholds, args.horizons, args.frameskip)

    torch.save(
        {
            "model": classifier.state_dict(),
            "input_dim": x_train.shape[1],
            "output_dim": len(args.horizons),
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "horizons": args.horizons,
            "frameskip": args.frameskip,
            "thresholds": thresholds,
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
        "latent_cache_metadata": latent_metadata,
        "label_cache_metadata": label_metadata,
        "normalizer": {"mean": x_mean.tolist(), "std": x_std.tolist()},
        "thresholds": thresholds.tolist(),
        "val": val_metrics,
        "test": test_metrics,
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"thresholds": thresholds.tolist(), "test": test_metrics}, indent=2))
    print(f"saved classifier to {output_dir / 'dense_reward_classifier.pt'}")


if __name__ == "__main__":
    main()
