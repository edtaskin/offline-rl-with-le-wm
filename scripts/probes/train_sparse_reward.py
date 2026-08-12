"""Train the PushT sparse reward classifier on frozen LeWM latents.

This is the dedicated entrypoint for the ``objective_met`` classifier used as
the sparse reward in imagined PPO. It keeps the same checkpoint format as
``scripts.probes.train_state --probes objective_met`` while exposing the
rollout-cache, target-FPR threshold, and hard-negative options that matter for
reward use.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes import train_dense_reward as dense  # noqa: E402
from scripts.probes import train_state as state_probe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/probes/pusht_sparse_reward")
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--imagined-cache", default=None)
    parser.add_argument("--imagined-val-cache", default=None)
    parser.add_argument("--imagined-test-cache", default=None)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
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
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--imagined-rollout-horizon", type=int, default=20)
    parser.add_argument("--include-imagined-rollouts", action="store_true")
    parser.add_argument("--imagined-fraction", type=float, default=0.5)
    parser.add_argument("--eval-imagined-rollouts", action="store_true")
    parser.add_argument("--imagined-eval-samples", type=int, default=100_000)
    parser.add_argument(
        "--threshold-policy",
        "--sparse-threshold-policy",
        choices=["f1", "target_fpr"],
        default="f1",
        help="Select threshold by validation F1 or highest recall with FPR <= target.",
    )
    parser.add_argument("--target-fpr", "--sparse-target-fpr", type=float, default=0.02)
    parser.add_argument(
        "--threshold-source",
        "--sparse-threshold-source",
        choices=["val", "imagined_val"],
        default="val",
        help="Validation split used to choose the deployed threshold.",
    )
    parser.add_argument("--hard-negative-mining", "--sparse-hard-negative-mining", action="store_true")
    parser.add_argument("--hard-negative-fraction", "--sparse-hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--hard-negative-epochs", "--sparse-hard-negative-epochs", type=int, default=10)
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--force-imagined-recache", action="store_true")
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM rollouts.",
    )
    parser.add_argument(
        "--sparse-imagined-sampling",
        choices=["uniform", "step_class_balanced"],
        default="uniform",
        help="How to subsample imagined train latents before mixing with GT latents.",
    )
    parser.add_argument(
        "--sparse-negative-sampling",
        choices=["natural", "mixed"],
        default="natural",
        help="Kept for experiment compatibility; BCE pos_weight still handles GT class imbalance.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    return parser.parse_args()


def default_latent_cache(output_dir: Path, args: argparse.Namespace) -> Path:
    existing_1m = state_probe.repo_path("models/probes/pusht_lewm_1M/latents.npz")
    if args.max_samples == 1_000_000 and existing_1m.exists():
        return existing_1m
    return output_dir / "latents.npz"


def objective_labels_for_rows(h5_path: Path, rows: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        state = h5["state"][:][rows.astype(np.int64)].astype(np.float32)
    return state_probe.targets(state, "objective_met", args).astype(np.float32)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, policy: str, target_fpr: float) -> float:
    y = y_true.reshape(-1).astype(np.float32)
    p = y_prob.reshape(-1)
    candidates = np.unique(np.quantile(p, np.linspace(0.001, 0.999, 200)))
    candidates = np.concatenate([[0.5], candidates])
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        metrics = state_probe.binary_confusion(y, p, float(threshold))
        if policy == "f1":
            score = metrics["f1"]
        elif policy == "target_fpr":
            fpr = metrics["fp"] / max(metrics["fp"] + metrics["tn"], 1.0)
            if fpr > target_fpr:
                continue
            score = metrics["recall"]
        else:
            raise ValueError(f"Unknown threshold policy: {policy}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = state_probe.classification_metrics(y_prob, y_true, threshold)
    conf = state_probe.binary_confusion(y_true, y_prob, threshold)
    metrics["false_positive_rate"] = conf["fp"] / max(conf["fp"] + conf["tn"], 1.0)
    return metrics


def choose_imagined_indices(arrays: dict[str, np.ndarray], y: np.ndarray, n: int, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 90_001)
    n = min(int(n), len(y))
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    if args.sparse_imagined_sampling == "uniform" or "model_step" not in arrays:
        return rng.choice(len(y), size=n, replace=len(y) < n).astype(np.int64)

    y_flat = y.reshape(-1).astype(bool)
    model_step = arrays["model_step"].astype(np.int64)
    buckets = []
    for step in np.unique(model_step):
        step_idx = np.flatnonzero(model_step == step)
        pos = step_idx[y_flat[step_idx]]
        neg = step_idx[~y_flat[step_idx]]
        if len(pos) == 0 or len(neg) == 0:
            buckets.append(step_idx)
            continue
        per_class = max(1, int(np.ceil(n / (2 * len(np.unique(model_step))))))
        buckets.append(rng.choice(pos, size=min(per_class, len(pos)), replace=False))
        buckets.append(rng.choice(neg, size=min(per_class, len(neg)), replace=False))
    idx = np.concatenate(buckets) if buckets else np.arange(len(y))
    if len(idx) >= n:
        return rng.choice(idx, size=n, replace=False).astype(np.int64)
    extra = rng.choice(len(y), size=n - len(idx), replace=len(y) < (n - len(idx)))
    return np.concatenate([idx, extra]).astype(np.int64)


def get_or_build_imagined(
    cache_path: Path,
    h5_path: Path,
    model: torch.nn.Module,
    source_rows: np.ndarray,
    num_samples: int,
    args: argparse.Namespace,
    split: str,
    seed_offset: int,
) -> tuple[dict[str, np.ndarray], dict]:
    if cache_path.exists() and not args.force_imagined_recache:
        arrays, metadata = dense.load_imagined_cache(cache_path)
        required = {"z", "rows"}
        if required.issubset(arrays):
            print(f"loaded imagined {split} cache: {cache_path}")
            return arrays, metadata
        print(f"imagined {split} cache missing {required - set(arrays)}, recaching: {cache_path}")

    dense_args = copy.copy(args)
    dense_args.horizons = [1]
    dense_args.keep_censored_negatives = True
    arrays = dense.build_imagined_train_cache(
        h5_path=h5_path,
        model=model,
        source_rows=source_rows,
        num_samples=num_samples,
        rollout_horizon=int(args.imagined_rollout_horizon),
        args=dense_args,
        split=split,
        seed_offset=seed_offset,
    )
    metadata = dense.imagined_metadata_expected(dense_args, int(args.imagined_rollout_horizon), num_samples, split)
    dense.save_imagined_cache(cache_path, arrays, metadata)
    print(f"saved imagined {split} cache: {cache_path}")
    return arrays, metadata


def train_one_classifier(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_for_threshold: np.ndarray,
    y_val_for_threshold: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, float, np.ndarray]:
    model, _ = state_probe.train_classifier(x_train, y_train, x_val_for_threshold, y_val_for_threshold, args, kind)
    train_prob = state_probe.predict_classifier(model, x_train, args.batch_size)
    val_prob = state_probe.predict_classifier(model, x_val_for_threshold, args.batch_size)
    threshold = select_threshold(y_val_for_threshold, val_prob, args.threshold_policy, args.target_fpr)

    if args.hard_negative_mining:
        negative = y_train.reshape(-1) < 0.5
        hard = np.flatnonzero(negative & (train_prob.reshape(-1) >= threshold))
        max_hard = int(round(len(x_train) * args.hard_negative_fraction))
        if len(hard) > max_hard:
            rng = np.random.default_rng(args.seed + (17 if kind == "linear" else 23))
            hard = rng.choice(hard, size=max_hard, replace=False)
        if len(hard) > 0:
            hn_args = copy.copy(args)
            hn_args.epochs = int(args.hard_negative_epochs)
            x_aug = np.concatenate([x_train, x_train[hard]], axis=0)
            y_aug = np.concatenate([y_train, y_train[hard]], axis=0)
            print(f"{kind} hard-negative mining: retraining with {len(hard)} duplicated negatives")
            model, _ = state_probe.train_classifier(x_aug, y_aug, x_val_for_threshold, y_val_for_threshold, hn_args, kind)
            val_prob = state_probe.predict_classifier(model, x_val_for_threshold, args.batch_size)
            threshold = select_threshold(y_val_for_threshold, val_prob, args.threshold_policy, args.target_fpr)
        else:
            print(f"{kind} hard-negative mining: no high-scoring negatives found")
    return model, threshold, val_prob


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.imagined_fraction <= 1.0:
        raise ValueError("--imagined-fraction must be between 0 and 1.")
    if not 0.0 <= args.target_fpr <= 1.0:
        raise ValueError("--target-fpr must be between 0 and 1.")

    output_dir = state_probe.repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = state_probe.dataset_path(args.cache_dir, args.dataset)
    cache_dir = state_probe.repo_path(args.cache_dir)
    latent_cache = state_probe.repo_path(args.latent_cache) if args.latent_cache else default_latent_cache(output_dir, args)
    imagined_cache = state_probe.repo_path(args.imagined_cache) if args.imagined_cache else output_dir / "imagined_rollouts_train.npz"
    imagined_val_cache = state_probe.repo_path(args.imagined_val_cache) if args.imagined_val_cache else output_dir / "imagined_rollouts_val.npz"
    imagined_test_cache = state_probe.repo_path(args.imagined_test_cache) if args.imagined_test_cache else output_dir / "imagined_rollouts_test.npz"

    if latent_cache.exists() and not args.force_recache:
        data, latent_metadata = state_probe.load_latent_cache(latent_cache)
        print(f"loaded latent cache: {latent_cache}")
    else:
        rows = state_probe.sample_rows(
            h5_path,
            args.max_samples,
            args.sample_block_size,
            state_probe.SplitConfig(seed=args.seed),
        )
        model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        data = state_probe.encode_rows(model, h5_path, rows, args.encode_batch_size, torch.device(args.device))
        latent_metadata = {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "max_samples": args.max_samples,
            "sample_block_size": args.sample_block_size,
            "seed": args.seed,
        }
        state_probe.save_latent_cache(latent_cache, data, latent_metadata)
        print(f"saved latent cache: {latent_cache}")

    y_train_gt = state_probe.targets(data["train"]["state"], "objective_met", args)
    y_val_gt = state_probe.targets(data["val"]["state"], "objective_met", args)
    y_test_gt = state_probe.targets(data["test"]["state"], "objective_met", args)

    z_train_raw = data["train"]["z"].astype(np.float32)
    train_source_counts = {"gt": int(len(z_train_raw)), "imagined": 0}
    imagined_metadata = imagined_val_metadata = imagined_test_metadata = None
    imagined_val = imagined_test = None
    rollout_model = None

    if args.include_imagined_rollouts and args.imagined_fraction > 0:
        n_train_total = len(z_train_raw)
        n_imagined = int(round(n_train_total * args.imagined_fraction))
        n_gt_keep = n_train_total - n_imagined
        rollout_model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        imagined, imagined_metadata = get_or_build_imagined(
            imagined_cache,
            h5_path,
            rollout_model,
            data["train"]["rows"],
            n_imagined,
            args,
            split="train",
            seed_offset=10_000,
        )
        imagined_y = objective_labels_for_rows(h5_path, imagined["rows"], args)
        imagined_idx = choose_imagined_indices(imagined, imagined_y, n_imagined, args)
        rng = np.random.default_rng(args.seed + 11)
        gt_keep_idx = rng.choice(n_train_total, size=n_gt_keep, replace=False) if n_gt_keep > 0 else np.empty((0,), dtype=np.int64)
        z_train_raw = np.concatenate([z_train_raw[gt_keep_idx], imagined["z"][imagined_idx].astype(np.float32)], axis=0)
        y_train_gt = np.concatenate([y_train_gt[gt_keep_idx], imagined_y[imagined_idx]], axis=0)
        train_source_counts = {"gt": int(n_gt_keep), "imagined": int(len(imagined_idx))}
        print(f"training mix: {n_gt_keep} GT latents + {len(imagined_idx)} imagined rollout latents")

    if args.eval_imagined_rollouts or args.threshold_source == "imagined_val":
        if rollout_model is None:
            rollout_model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
        imagined_val, imagined_val_metadata = get_or_build_imagined(
            imagined_val_cache,
            h5_path,
            rollout_model,
            data["val"]["rows"],
            args.imagined_eval_samples,
            args,
            split="val",
            seed_offset=20_000,
        )
        imagined_test, imagined_test_metadata = get_or_build_imagined(
            imagined_test_cache,
            h5_path,
            rollout_model,
            data["test"]["rows"],
            args.imagined_eval_samples,
            args,
            split="test",
            seed_offset=30_000,
        )

    z_train, (z_val, z_test), z_mean, z_std = state_probe.standardize(
        z_train_raw.astype(np.float32),
        data["val"]["z"].astype(np.float32),
        data["test"]["z"].astype(np.float32),
    )
    if args.threshold_source == "imagined_val":
        if imagined_val is None:
            raise RuntimeError("threshold_source=imagined_val requires imagined validation cache")
        z_val_threshold = ((imagined_val["z"].astype(np.float32) - z_mean) / z_std).astype(np.float32)
        y_val_threshold = objective_labels_for_rows(h5_path, imagined_val["rows"], args)
    else:
        z_val_threshold = z_val
        y_val_threshold = y_val_gt

    feature_dir = output_dir / "objective_met"
    feature_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": vars(args),
        "feature_normalizer": {"mean": z_mean.tolist(), "std": z_std.tolist()},
        "latent_cache": str(latent_cache),
        "imagined_cache": str(imagined_cache) if args.include_imagined_rollouts else None,
        "imagined_val_cache": str(imagined_val_cache) if (args.eval_imagined_rollouts or args.threshold_source == "imagined_val") else None,
        "imagined_test_cache": str(imagined_test_cache) if args.eval_imagined_rollouts else None,
        "latent_cache_metadata": latent_metadata,
        "imagined_cache_metadata": imagined_metadata,
        "imagined_val_cache_metadata": imagined_val_metadata,
        "imagined_test_cache_metadata": imagined_test_metadata,
        "train_source_counts": train_source_counts,
        "probes": {"objective_met": {}},
    }
    class_counts = {
        "train_positive": int(y_train_gt.sum()),
        "train_negative": int(len(y_train_gt) - y_train_gt.sum()),
        "val_positive": int(y_val_gt.sum()),
        "val_negative": int(len(y_val_gt) - y_val_gt.sum()),
        "test_positive": int(y_test_gt.sum()),
        "test_negative": int(len(y_test_gt) - y_test_gt.sum()),
    }

    for kind, path_name in (("linear", "linear_probe.pt"), ("mlp", "mlp_probe.pt")):
        model, threshold, _ = train_one_classifier(kind, z_train, y_train_gt, z_val_threshold, y_val_threshold, args)
        test_prob = state_probe.predict_classifier(model, z_test, args.batch_size)
        metrics = classification_metrics(test_prob, y_test_gt, threshold)
        payload = {
            "model": model.state_dict(),
            "input_dim": z_train.shape[1],
            "output_dim": y_train_gt.shape[1],
            "x_mean": z_mean,
            "x_std": z_std,
            "threshold": threshold,
            "task": "binary_classification",
        }
        if kind == "mlp":
            payload["hidden_dim"] = args.mlp_hidden
            payload["depth"] = args.mlp_depth
        torch.save(payload, feature_dir / path_name)
        results["probes"]["objective_met"][kind] = metrics
        print(f"objective_met {kind}: {metrics}")

    if imagined_val is not None and imagined_test is not None:
        for split_name, arrays in (("imagined_val", imagined_val), ("imagined_test", imagined_test)):
            z_split = ((arrays["z"].astype(np.float32) - z_mean) / z_std).astype(np.float32)
            y_split = objective_labels_for_rows(h5_path, arrays["rows"], args)
            split_metrics = {}
            for kind, path_name in (("linear", "linear_probe.pt"), ("mlp", "mlp_probe.pt")):
                checkpoint = torch.load(feature_dir / path_name, map_location="cpu", weights_only=False)
                if kind == "linear":
                    model = state_probe.LinearClassifier(checkpoint["input_dim"], checkpoint["output_dim"])
                else:
                    model = state_probe.ProbeMLP(
                        checkpoint["input_dim"],
                        checkpoint["output_dim"],
                        checkpoint["hidden_dim"],
                        checkpoint["depth"],
                    )
                model.load_state_dict(checkpoint["model"])
                prob = state_probe.predict_classifier(model.eval(), z_split, args.batch_size)
                split_metrics[kind] = classification_metrics(prob, y_split, float(checkpoint["threshold"]))
            results[split_name] = split_metrics

    results["probes"]["objective_met"]["class_counts"] = class_counts
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"class_counts": class_counts, "train_source_counts": train_source_counts}, indent=2))
    print(f"wrote sparse reward classifier to {feature_dir}")


if __name__ == "__main__":
    main()
