"""Evaluate PushT LeWM imagined rollouts with trained linear probes.

For each sampled trajectory, this script:
  1. encodes a short context from ground-truth frames,
  2. rolls LeWM forward using the ground-truth action blocks,
  3. decodes each imagined latent with the saved linear probes,
  4. compares decoded physical quantities to ground-truth states over time.

The default paths assume this script is run from the top-level wrapper repo:
dataset/checkpoint under le-wm/models, probes under models/probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch
from torch import nn
from einops import rearrange


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
REGRESSION_FEATURES = (
    "agent_pos",
    "block_pos",
    "block_angle",
    "block_rel_objective",
    "block_rel_agent",
)
CLASSIFICATION_FEATURES = ("objective_met",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm_200k")
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=3,
        help="Number of GT frames used to initialize rollout; 3 matches the trained PushT LeWM history.",
    )
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
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM.",
    )
    return parser.parse_args()


def preprocess_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 4, 2, 3).div_(255.0)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


class LinearProbe:
    def __init__(self, path: Path):
        raw = np.load(path)
        self.weights = raw["weights"].astype(np.float32)
        self.x_mean = raw["x_mean"].astype(np.float32)
        self.x_std = raw["x_std"].astype(np.float32)
        self.y_mean = raw["y_mean"].astype(np.float32)
        self.y_std = raw["y_std"].astype(np.float32)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        original_shape = z.shape[:-1]
        x = z.reshape(-1, z.shape[-1]).astype(np.float32)
        x = (x - self.x_mean) / self.x_std
        x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
        y = x_aug @ self.weights
        y = y * self.y_std + self.y_mean
        return y.reshape(*original_shape, -1)


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


class MLPProbe:
    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.x_mean = payload["x_mean"].astype(np.float32)
        self.x_std = payload["x_std"].astype(np.float32)
        self.y_mean = payload["y_mean"].astype(np.float32)
        self.y_std = payload["y_std"].astype(np.float32)
        self.model = ProbeMLP(
            payload["input_dim"],
            payload["output_dim"],
            payload["hidden_dim"],
            payload["depth"],
        )
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, z: np.ndarray) -> np.ndarray:
        original_shape = z.shape[:-1]
        x = z.reshape(-1, z.shape[-1]).astype(np.float32)
        x = (x - self.x_mean) / self.x_std
        y = self.model(torch.from_numpy(x).float()).numpy()
        y = y * self.y_std + self.y_mean
        return y.reshape(*original_shape, -1)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryClassifierProbe:
    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.x_mean = payload["x_mean"].astype(np.float32)
        self.x_std = payload["x_std"].astype(np.float32)
        self.threshold = float(payload["threshold"])
        if "hidden_dim" in payload:
            self.model = ProbeMLP(
                payload["input_dim"],
                payload["output_dim"],
                payload["hidden_dim"],
                payload["depth"],
            )
        else:
            self.model = LinearClassifier(payload["input_dim"], payload["output_dim"])
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, z: np.ndarray) -> np.ndarray:
        original_shape = z.shape[:-1]
        x = z.reshape(-1, z.shape[-1]).astype(np.float32)
        x = (x - self.x_mean) / self.x_std
        logits = self.model(torch.from_numpy(x).float())
        prob = torch.sigmoid(logits).numpy()
        return prob.reshape(*original_shape, -1)


def load_probes(
    probe_dir: Path,
    probe_kind: str,
) -> tuple[dict[str, LinearProbe | MLPProbe], dict[str, BinaryClassifierProbe]]:
    probes = {}
    for feature in REGRESSION_FEATURES:
        path = probe_dir / feature / ("linear_probe.npz" if probe_kind == "linear" else "mlp_probe.pt")
        if not path.exists():
            raise FileNotFoundError(f"Missing {probe_kind} probe: {path}")
        probes[feature] = LinearProbe(path) if probe_kind == "linear" else MLPProbe(path)

    classifiers = {}
    for feature in CLASSIFICATION_FEATURES:
        path = probe_dir / feature / f"{probe_kind}_probe.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {probe_kind} classifier probe: {path}")
        classifiers[feature] = BinaryClassifierProbe(path)
    return probes, classifiers


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bsz = len(starts)
    action_steps = context_steps + horizon - 1
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    future_pixels = np.empty((bsz, horizon, 224, 224, 3), dtype=np.uint8)
    action_blocks = np.empty((bsz, action_steps, frameskip * 2), dtype=np.float32)
    future_states = np.empty((bsz, horizon, 7), dtype=np.float32)

    for i, (_, start) in enumerate(starts):
        context_idx = start + np.arange(context_steps) * frameskip
        future_idx = start + (context_steps + np.arange(horizon)) * frameskip
        context_pixels[i] = h5["pixels"][context_idx]
        future_pixels[i] = h5["pixels"][future_idx]
        future_states[i] = h5["state"][future_idx]
        for t in range(action_steps):
            a0 = start + t * frameskip
            raw_action = h5["action"][a0 : a0 + frameskip].astype(np.float32)
            if normalize_actions:
                raw_action = (raw_action - action_mean) / action_std
            action_blocks[i, t] = raw_action.reshape(-1)

    return context_pixels, future_pixels, action_blocks, future_states


@torch.inference_mode()
def rollout_embeddings(
    model: torch.nn.Module,
    context_pixels: np.ndarray,
    action_blocks: np.ndarray,
    horizon: int,
    history_size: int,
    device: torch.device,
) -> np.ndarray:
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

    pred_emb = torch.stack(emb_list[context_steps:], dim=1)
    return pred_emb.detach().cpu().float().numpy()


@torch.inference_mode()
def encode_future_embeddings(
    model: torch.nn.Module,
    future_pixels: np.ndarray,
    encode_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    bsz, horizon = future_pixels.shape[:2]
    flat_pixels = future_pixels.reshape(bsz * horizon, *future_pixels.shape[2:])
    chunks = []
    for start in range(0, len(flat_pixels), encode_batch_size):
        pixel_chunk = flat_pixels[start : start + encode_batch_size]
        pixels = preprocess_pixels(pixel_chunk[:, None], device)
        emb = model.encode({"pixels": pixels})["emb"][:, 0]
        chunks.append(emb.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).reshape(bsz, horizon, -1)


def probe_predictions(probes: dict[str, LinearProbe | MLPProbe], z: np.ndarray) -> dict[str, np.ndarray]:
    return {feature: probe(z) for feature, probe in probes.items()}


def classifier_predictions(probes: dict[str, BinaryClassifierProbe], z: np.ndarray) -> dict[str, np.ndarray]:
    return {feature: probe(z) for feature, probe in probes.items()}


def angle_delta(angle: np.ndarray, reference: float | np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle - reference), np.cos(angle - reference))


def rotate_2d(vec: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    x = vec[..., 0]
    y = vec[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def ground_truth_targets(states: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    objective_pos = np.array([args.objective_x, args.objective_y], dtype=np.float32)
    objective_angle = float(args.objective_angle)
    rel_objective_angle = angle_delta(states[..., 4], objective_angle)
    rel_objective_xy = rotate_2d(states[..., 2:4] - objective_pos, -objective_angle)
    rel_agent_xy = rotate_2d(states[..., 0:2] - states[..., 2:4], -states[..., 4])
    pos_err = np.linalg.norm(states[..., 2:4] - objective_pos, axis=-1)
    angle_err = np.abs(rel_objective_angle)

    return {
        "agent_pos": states[..., 0:2],
        "block_pos": states[..., 2:4],
        "block_angle": np.stack([np.sin(states[..., 4]), np.cos(states[..., 4])], axis=-1),
        "block_rel_objective": np.concatenate(
            [
                rel_objective_xy,
                np.sin(rel_objective_angle)[..., None],
                np.cos(rel_objective_angle)[..., None],
            ],
            axis=-1,
        ),
        "block_rel_agent": np.concatenate(
            [rel_agent_xy, np.sin(states[..., 4])[..., None], np.cos(states[..., 4])[..., None]],
            axis=-1,
        ),
        "objective_met": (
            (pos_err <= args.objective_pos_tol) & (angle_err <= args.objective_angle_tol)
        )[..., None].astype(np.float32),
    }


def angle_diff_rad(pred_sincos: np.ndarray, true_angle: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(pred_sincos, axis=-1), 1e-8)
    pred_angle = np.arctan2(pred_sincos[..., 0] / norm, pred_sincos[..., 1] / norm)
    return np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle))


def sincos_diff_rad(pred_sincos: np.ndarray, true_sincos: np.ndarray) -> np.ndarray:
    pred_norm = pred_sincos / np.maximum(np.linalg.norm(pred_sincos, axis=-1, keepdims=True), 1e-8)
    true_norm = true_sincos / np.maximum(np.linalg.norm(true_sincos, axis=-1, keepdims=True), 1e-8)
    pred_angle = np.arctan2(pred_norm[..., 0], pred_norm[..., 1])
    true_angle = np.arctan2(true_norm[..., 0], true_norm[..., 1])
    return np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle))


def compute_curves(pred: dict[str, np.ndarray], states: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    curves = {}
    gt = ground_truth_targets(states, args)
    agent_err = pred["agent_pos"] - gt["agent_pos"]
    block_err = pred["block_pos"] - gt["block_pos"]
    angle_err = sincos_diff_rad(pred["block_angle"], gt["block_angle"])
    rel_objective_xy_err = pred["block_rel_objective"][..., :2] - gt["block_rel_objective"][..., :2]
    rel_objective_angle_err = sincos_diff_rad(
        pred["block_rel_objective"][..., 2:4], gt["block_rel_objective"][..., 2:4]
    )
    rel_agent_xy_err = pred["block_rel_agent"][..., :2] - gt["block_rel_agent"][..., :2]
    rel_agent_angle_err = sincos_diff_rad(
        pred["block_rel_agent"][..., 2:4], gt["block_rel_agent"][..., 2:4]
    )

    curves["agent_pos_rmse_px"] = np.sqrt(np.mean(agent_err**2, axis=(0, 2)))
    curves["agent_pos_mae_px"] = np.mean(np.abs(agent_err), axis=(0, 2))
    curves["block_pos_rmse_px"] = np.sqrt(np.mean(block_err**2, axis=(0, 2)))
    curves["block_pos_mae_px"] = np.mean(np.abs(block_err), axis=(0, 2))
    curves["block_angle_rmse_deg"] = np.degrees(np.sqrt(np.mean(angle_err**2, axis=0)))
    curves["block_angle_mae_deg"] = np.degrees(np.mean(np.abs(angle_err), axis=0))
    curves["block_rel_objective_xy_rmse_px"] = np.sqrt(np.mean(rel_objective_xy_err**2, axis=(0, 2)))
    curves["block_rel_objective_xy_mae_px"] = np.mean(np.abs(rel_objective_xy_err), axis=(0, 2))
    curves["block_rel_objective_angle_rmse_deg"] = np.degrees(
        np.sqrt(np.mean(rel_objective_angle_err**2, axis=0))
    )
    curves["block_rel_objective_angle_mae_deg"] = np.degrees(np.mean(np.abs(rel_objective_angle_err), axis=0))
    curves["block_rel_agent_xy_rmse_px"] = np.sqrt(np.mean(rel_agent_xy_err**2, axis=(0, 2)))
    curves["block_rel_agent_xy_mae_px"] = np.mean(np.abs(rel_agent_xy_err), axis=(0, 2))
    curves["block_rel_agent_angle_rmse_deg"] = np.degrees(np.sqrt(np.mean(rel_agent_angle_err**2, axis=0)))
    curves["block_rel_agent_angle_mae_deg"] = np.degrees(np.mean(np.abs(rel_agent_angle_err), axis=0))
    return curves


def binary_curve_metrics(
    prob: np.ndarray,
    states: np.ndarray,
    args: argparse.Namespace,
    threshold: float,
) -> dict[str, np.ndarray]:
    y = ground_truth_targets(states, args)["objective_met"][..., 0].astype(bool)
    p = prob[..., 0]
    pred = p >= threshold
    tp = np.logical_and(pred, y).sum(axis=0).astype(np.float32)
    tn = np.logical_and(~pred, ~y).sum(axis=0).astype(np.float32)
    fp = np.logical_and(pred, ~y).sum(axis=0).astype(np.float32)
    fn = np.logical_and(~pred, y).sum(axis=0).astype(np.float32)
    recall = tp / np.maximum(tp + fn, 1.0)
    specificity = tn / np.maximum(tn + fp, 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    return {
        "objective_met_positive_rate": y.mean(axis=0),
        "objective_met_mean_probability": p.mean(axis=0),
        "objective_met_predicted_rate": pred.mean(axis=0),
        "objective_met_accuracy": (tp + tn) / np.maximum(tp + tn + fp + fn, 1.0),
        "objective_met_balanced_accuracy": 0.5 * (recall + specificity),
        "objective_met_precision": precision,
        "objective_met_recall": recall,
        "objective_met_f1": f1,
    }


def latent_curve(pred_emb: np.ndarray, gt_emb: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred_emb - gt_emb) ** 2, axis=(0, 2)))


def latent_cosine_curve(pred_emb: np.ndarray, gt_emb: np.ndarray) -> np.ndarray:
    numerator = np.sum(pred_emb * gt_emb, axis=-1)
    pred_norm = np.linalg.norm(pred_emb, axis=-1)
    gt_norm = np.linalg.norm(gt_emb, axis=-1)
    cosine = numerator / np.maximum(pred_norm * gt_norm, 1e-8)
    return np.mean(cosine, axis=0)


def save_csv(path: Path, steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_step", "env_step"] + sorted(curves)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, step in enumerate(steps):
            row = {"model_step": int(i + 1), "env_step": int(step)}
            row.update({key: float(value[i]) for key, value in curves.items()})
            writer.writerow(row)


def plot_curves(
    path: Path,
    env_steps: np.ndarray,
    imagined: dict[str, np.ndarray],
    encoded_gt: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    specs = [
        ("agent_pos_rmse_px", "Agent Position RMSE (px)"),
        ("block_pos_rmse_px", "Block Position RMSE (px)"),
        ("block_angle_rmse_deg", "Block Angle RMSE (deg)"),
    ]
    for ax, (key, title) in zip(axes, specs):
        ax.plot(env_steps, imagined[key], label="imagined latent", linewidth=2)
        ax.plot(env_steps, encoded_gt[key], label="encoded GT latent", linestyle="--", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Environment steps after context")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Error")
    axes[-1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_relative_curves(
    path: Path,
    env_steps: np.ndarray,
    imagined: dict[str, np.ndarray],
    encoded_gt: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [
        ("block_rel_objective_xy_rmse_px", "T Relative to Objective XY RMSE (px)"),
        ("block_rel_objective_angle_rmse_deg", "T Relative to Objective Angle RMSE (deg)"),
        ("block_rel_agent_xy_rmse_px", "T Relative to Agent XY RMSE (px)"),
        ("block_rel_agent_angle_rmse_deg", "T Relative to Agent Angle RMSE (deg)"),
    ]
    for ax, (key, title) in zip(axes.flat, specs):
        ax.plot(env_steps, imagined[key], label="imagined latent", linewidth=2)
        ax.plot(env_steps, encoded_gt[key], label="encoded GT latent", linestyle="--", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Environment steps after context")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Error")
    axes[1, 0].set_ylabel("Error")
    axes[0, 1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_objective_met_curves(
    path: Path,
    env_steps: np.ndarray,
    imagined: dict[str, np.ndarray],
    encoded_gt: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    specs = [
        ("objective_met_mean_probability", "Mean P(objective met)"),
        ("objective_met_f1", "Objective Met F1"),
        ("objective_met_balanced_accuracy", "Objective Met Balanced Accuracy"),
    ]
    for ax, (key, title) in zip(axes, specs):
        ax.plot(env_steps, imagined[key], label="imagined latent", linewidth=2)
        ax.plot(env_steps, encoded_gt[key], label="encoded GT latent", linestyle="--", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Environment steps after context")
        ax.grid(True, alpha=0.25)
        if key != "objective_met_mean_probability":
            ax.set_ylim(0.0, 1.05)
    axes[-1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_latent_curve(path: Path, env_steps: np.ndarray, latent_rmse: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(env_steps, latent_rmse, linewidth=2)
    ax.set_title("Predicted vs Encoded GT Latent RMSE")
    ax.set_xlabel("Environment steps after context")
    ax.set_ylabel("Latent RMSE")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_latent_similarity(
    path: Path,
    env_steps: np.ndarray,
    latent_rmse: np.ndarray,
    latent_cosine: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(env_steps, latent_rmse, linewidth=2)
    axes[0].set_title("Predicted vs Encoded GT Latent RMSE")
    axes[0].set_xlabel("Environment steps after context")
    axes[0].set_ylabel("Latent RMSE")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(env_steps, latent_cosine, linewidth=2)
    axes[1].set_title("Predicted vs Encoded GT Latent Cosine")
    axes[1].set_xlabel("Environment steps after context")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or f"models/rollout_probe/pusht_lewm_200k_{args.probe_kind}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    probes, classifier_probes = load_probes(Path(args.probe_dir), args.probe_kind)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=args.checkpoint_cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", 3))

    pred_embs = []
    gt_embs = []
    states = []
    sampled = []

    with h5py.File(args.dataset_path, "r") as h5:
        action_mean, action_std = action_stats(h5)
        starts = sample_starts(
            h5=h5,
            num_trajectories=args.num_trajectories,
            context_steps=args.context_steps,
            horizon=args.horizon,
            frameskip=args.frameskip,
            seed=args.seed,
        )
        for start in range(0, len(starts), args.batch_size):
            batch_starts = starts[start : start + args.batch_size]
            context_pixels, future_pixels, action_blocks, future_states = load_batch(
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
            gt_emb = encode_future_embeddings(
                model,
                future_pixels,
                encode_batch_size=args.encode_batch_size,
                device=device,
            )
            pred_embs.append(pred_emb)
            gt_embs.append(gt_emb)
            states.append(future_states)
            sampled.extend(batch_starts)
            print(f"processed {len(sampled)}/{len(starts)} trajectories")

    pred_emb = np.concatenate(pred_embs, axis=0)
    gt_emb = np.concatenate(gt_embs, axis=0)
    future_states = np.concatenate(states, axis=0)

    imagined_pred = probe_predictions(probes, pred_emb)
    encoded_gt_pred = probe_predictions(probes, gt_emb)
    imagined_class_pred = classifier_predictions(classifier_probes, pred_emb)
    encoded_gt_class_pred = classifier_predictions(classifier_probes, gt_emb)
    imagined_curves = compute_curves(imagined_pred, future_states, args)
    encoded_gt_curves = compute_curves(encoded_gt_pred, future_states, args)
    objective_threshold = classifier_probes["objective_met"].threshold
    imagined_curves.update(
        binary_curve_metrics(
            imagined_class_pred["objective_met"],
            future_states,
            args,
            threshold=objective_threshold,
        )
    )
    encoded_gt_curves.update(
        binary_curve_metrics(
            encoded_gt_class_pred["objective_met"],
            future_states,
            args,
            threshold=objective_threshold,
        )
    )
    latent_rmse = latent_curve(pred_emb, gt_emb)
    latent_cosine = latent_cosine_curve(pred_emb, gt_emb)

    env_steps = args.frameskip * np.arange(1, args.horizon + 1)
    all_curves = {
        **{f"imagined_{k}": v for k, v in imagined_curves.items()},
        **{f"encoded_gt_{k}": v for k, v in encoded_gt_curves.items()},
        "latent_rmse": latent_rmse,
        "latent_cosine": latent_cosine,
    }

    save_csv(output_dir / "rollout_probe_curves.csv", env_steps, all_curves)
    plot_curves(output_dir / "rollout_probe_rmse.png", env_steps, imagined_curves, encoded_gt_curves)
    plot_relative_curves(output_dir / "rollout_probe_relative_rmse.png", env_steps, imagined_curves, encoded_gt_curves)
    plot_objective_met_curves(
        output_dir / "rollout_probe_objective_met.png",
        env_steps,
        imagined_curves,
        encoded_gt_curves,
    )
    plot_latent_curve(output_dir / "latent_rmse.png", env_steps, latent_rmse)
    plot_latent_similarity(output_dir / "latent_similarity.png", env_steps, latent_rmse, latent_cosine)
    np.savez_compressed(
        output_dir / "rollout_probe_arrays.npz",
        pred_emb=pred_emb,
        gt_emb=gt_emb,
        future_states=future_states,
        env_steps=env_steps,
        sampled=np.asarray(sampled, dtype=np.int64),
        imagined_objective_met_probability=imagined_class_pred["objective_met"],
        encoded_gt_objective_met_probability=encoded_gt_class_pred["objective_met"],
        **all_curves,
    )

    summary = {
        "config": vars(args),
        "num_trajectories": len(sampled),
        "history_size": history_size,
        "objective_met_threshold": objective_threshold,
        "action_normalization": {
            "enabled": not args.no_normalize_actions,
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
        "final_step": {
            key: float(value[-1])
            for key, value in all_curves.items()
            if key.endswith("_rmse_px")
            or key.endswith("_rmse_deg")
            or key.endswith("_xy_rmse_px")
            or key.endswith("_angle_rmse_deg")
            or key.endswith("_f1")
            or key.endswith("_balanced_accuracy")
            or key.endswith("_mean_probability")
            or key.endswith("_positive_rate")
            or key in ("latent_rmse", "latent_cosine")
        },
        "mean_over_horizon": {
            key: float(np.mean(value))
            for key, value in all_curves.items()
            if key.endswith("_rmse_px")
            or key.endswith("_rmse_deg")
            or key.endswith("_xy_rmse_px")
            or key.endswith("_angle_rmse_deg")
            or key.endswith("_f1")
            or key.endswith("_balanced_accuracy")
            or key.endswith("_mean_probability")
            or key.endswith("_positive_rate")
            or key in ("latent_rmse", "latent_cosine")
        },
    }
    with (output_dir / "rollout_probe_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["final_step"], indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
