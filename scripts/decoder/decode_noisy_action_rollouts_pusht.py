"""Visualize PushT LeWM rollouts under noisy expert actions.

This extends ``decode_rollouts_pusht.py`` from pure expert futures to a
counterfactual rollout: initialize from expert context frames, add noise to the
future expert actions, step the real PushT simulator from the recorded state,
and compare those simulated frames against decoded LeWM imagination driven by
the same noisy action blocks.

The expert dataset stores actions in the stable-world-model relative PushT
format, so the simulator path must use ``relative=True``.

Relative paths are resolved from the top-level wrapper repo.
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
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder.decode_rollouts_pusht import (
    MetricAccumulator,
    action_stats,
    decode_embeddings,
    image_metric_parts,
    load_decoder,
    parse_grid_steps,
    pixels_to_tensor,
    plot_metric_group,
    preprocess_pixels,
    sample_starts,
    save_metrics_csv,
    tensor_image_to_uint8,
)
from scripts.probes import probe_rollouts_pusht as probe_eval
from src.envs import PUSHT_FIXED_TARGET_POSE, make_pusht_env


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["visualize", "evaluate_rollouts"],
        default="visualize",
        help="Run the visual decoder comparison or the probe-based noisy rollout evaluation.",
    )
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--decoder-checkpoint", default="models/latent_decoder/pusht_lewm/decoder_best.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--video-format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--grid-steps", default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--no-grids", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    parser.add_argument(
        "--noise-stds",
        default="0,0.025,0.05,0.1,0.2",
        help="Comma-separated action-noise stds/half-widths for evaluate_rollouts.",
    )
    parser.add_argument(
        "--action-noise-mode",
        choices=["gaussian", "uniform"],
        default="gaussian",
    )
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm_1M")
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
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
        help="Disable dataset z-score normalization before feeding noisy action blocks to LeWM.",
    )
    return parser.parse_args()


def parse_noise_stds(value: str) -> list[float]:
    noise_stds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not noise_stds:
        raise ValueError("--noise-stds must contain at least one value")
    if any(std < 0 for std in noise_stds):
        raise ValueError("--noise-stds values must be non-negative")
    return noise_stds


def make_goal_state(goal_pose: np.ndarray, state_dim: int) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(3)
    if state_dim not in (5, 7):
        raise ValueError(f"expected 5D or 7D PushT states, got {state_dim}D")
    state = np.array([256.0, 256.0, *goal_pose], dtype=np.float64)
    if state_dim == 7:
        state = np.concatenate((state, np.zeros(2, dtype=np.float64)))
    return state


def make_noise(shape: tuple[int, ...], args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if args.action_noise_std < 0:
        raise ValueError("--action-noise-std must be non-negative")
    if args.action_noise_mode == "gaussian":
        return rng.normal(0.0, args.action_noise_std, size=shape).astype(np.float32)
    return rng.uniform(-args.action_noise_std, args.action_noise_std, size=shape).astype(np.float32)


def build_noisy_action_blocks(
    h5: h5py.File,
    starts: list[tuple[int, int]],
    *,
    context_steps: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalize_actions: bool,
    action_low: np.ndarray,
    action_high: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy LeWM action blocks and noisy future env actions.

    Action blocks before the latest context frame stay clean because they explain
    already-observed context frames. Noise starts at block ``context_steps - 1``.
    """
    bsz = len(starts)
    action_steps = context_steps + horizon - 1
    raw_blocks = np.empty((bsz, action_steps, frameskip, 2), dtype=np.float32)

    for i, (_, start) in enumerate(starts):
        for t in range(action_steps):
            a0 = start + t * frameskip
            raw_blocks[i, t] = h5["action"][a0 : a0 + frameskip].astype(np.float32)

    future = raw_blocks[:, context_steps - 1 :].reshape(bsz, horizon * frameskip, 2)
    future_noisy = np.clip(
        future + make_noise(future.shape, args, rng),
        action_low.reshape(1, 1, 2),
        action_high.reshape(1, 1, 2),
    ).astype(np.float32)
    raw_blocks[:, context_steps - 1 :] = future_noisy.reshape(bsz, horizon, frameskip, 2)

    blocks = raw_blocks
    if normalize_actions:
        blocks = (blocks - action_mean.reshape(1, 1, 1, 2)) / action_std.reshape(1, 1, 1, 2)
    return blocks.reshape(bsz, action_steps, frameskip * 2).astype(np.float32), future_noisy


def load_context_future_state(
    h5: h5py.File,
    starts: list[tuple[int, int]],
    *,
    context_steps: int,
    horizon: int,
    frameskip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz = len(starts)
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    expert_future_pixels = np.empty((bsz, horizon, 224, 224, 3), dtype=np.uint8)
    init_states = np.empty((bsz, h5["state"].shape[1]), dtype=np.float64)

    for i, (_, start) in enumerate(starts):
        context_idx = start + np.arange(context_steps) * frameskip
        future_idx = start + (context_steps + np.arange(horizon)) * frameskip
        context_pixels[i] = h5["pixels"][context_idx]
        expert_future_pixels[i] = h5["pixels"][future_idx]
        init_states[i] = h5["state"][start + (context_steps - 1) * frameskip]
    return context_pixels, expert_future_pixels, init_states


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


def rollout_simulator_images(
    init_states: np.ndarray,
    noisy_actions: np.ndarray,
    *,
    horizon: int,
    frameskip: int,
    resolution: int = 224,
) -> np.ndarray:
    pixels, _ = rollout_simulator(init_states, noisy_actions, horizon=horizon, frameskip=frameskip, resolution=resolution)
    return pixels


def rollout_simulator(
    init_states: np.ndarray,
    noisy_actions: np.ndarray,
    *,
    horizon: int,
    frameskip: int,
    resolution: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    bsz = len(init_states)
    pixels = np.empty((bsz, horizon, resolution, resolution, 3), dtype=np.uint8)
    states = np.empty((bsz, horizon, init_states.shape[1]), dtype=np.float32)
    goal_pose = np.asarray(PUSHT_FIXED_TARGET_POSE, dtype=np.float64)
    goal_state = make_goal_state(goal_pose, init_states.shape[1])

    env = make_pusht_env(
        render_mode="rgb_array",
        render_obs=False,
        resolution=resolution,
        relative=True,
        disable_env_checker=True,
        sync_goal_pose=False,
        align_sampled_goal_to_fixed_target=False,
        max_episode_steps=horizon * frameskip + 1,
    )
    try:
        for i in range(bsz):
            env.reset(options={"state": init_states[i], "goal_state": goal_state})
            unwrapped = env.unwrapped
            unwrapped.goal_pose = goal_pose.copy()
            unwrapped.goal_state = goal_state.copy()
            done = False
            last_frame = np.asarray(unwrapped.render(), dtype=np.uint8)
            for step in range(horizon):
                for j in range(frameskip):
                    if not done:
                        _, _, terminated, truncated, _ = env.step(
                            noisy_actions[i, step * frameskip + j]
                        )
                        done = bool(terminated or truncated)
                        unwrapped.goal_pose = goal_pose.copy()
                        unwrapped.goal_state = goal_state.copy()
                        last_frame = np.asarray(unwrapped.render(), dtype=np.uint8)
                pixels[i, step] = last_frame
                states[i, step] = np.asarray(unwrapped._get_obs(), dtype=np.float32)[: init_states.shape[1]]
    finally:
        env.close()
    return pixels, states


def add_label(image: np.ndarray, label: str, header_height: int = 24) -> np.ndarray:
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + header_height), "white")
    canvas.paste(Image.fromarray(image), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return np.asarray(canvas)


def comparison_frame(real_noisy: torch.Tensor, decoded: torch.Tensor, env_step: int) -> np.ndarray:
    left = add_label(tensor_image_to_uint8(real_noisy), f"sim GT+noise, +{env_step} env steps")
    right = add_label(tensor_image_to_uint8(decoded), f"decoded LeWM, +{env_step} env steps")
    divider = np.zeros((left.shape[0], 2, 3), dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def save_video(
    path: Path,
    real_noisy: torch.Tensor,
    decoded: torch.Tensor,
    env_steps: np.ndarray,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        comparison_frame(real_noisy[i], decoded[i], int(env_steps[i]))
        for i in range(len(env_steps))
    ]
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps)
    else:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def save_timestep_grid(
    path: Path,
    real_noisy: torch.Tensor,
    decoded: torch.Tensor,
    env_steps: np.ndarray,
    frameskip: int,
) -> None:
    selected = (env_steps // frameskip) - 1
    tile = 224
    left_margin = 92
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
        canvas.paste(Image.fromarray(tensor_image_to_uint8(real_noisy[int(idx)])), (x, y_real))
        canvas.paste(Image.fromarray(tensor_image_to_uint8(decoded[int(idx)])), (x, y_decoded))

    draw.text((8, top_margin + pad + tile // 2 - 8), "sim noise", fill=(0, 0, 0))
    draw.text((8, top_margin + 2 * pad + tile + tile // 2 - 8), "decoded", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_noisy_action_csv(path: Path, sampled: list[tuple[int, int]], noisy_actions: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trajectory", "episode", "start", "env_step", "action_x", "action_y"],
        )
        writer.writeheader()
        for traj_idx, ((ep, start), actions) in enumerate(zip(sampled, noisy_actions)):
            for t, action in enumerate(actions):
                writer.writerow(
                    {
                        "trajectory": traj_idx,
                        "episode": int(ep),
                        "start": int(start),
                        "env_step": int(t + 1),
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                    }
                )


def save_noisy_metric_plots(output_dir: Path, env_steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    comparisons = [
        ("imagined_vs_noisy_real", "decoded LeWM vs noisy sim"),
        ("noisy_real_vs_expert", "noisy sim vs expert"),
        ("imagined_vs_expert", "decoded LeWM vs expert"),
    ]
    plot_metric_group(
        output_dir / "noisy_action_image_errors.png",
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
        output_dir / "noisy_action_mask_iou.png",
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
        output_dir / "noisy_action_centroid_errors.png",
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


def flatten_curves_by_noise(results: dict[float, dict[str, dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    arrays = {}
    for noise_std, groups in results.items():
        noise_key = f"noise_{noise_std:g}".replace(".", "p")
        for group_name, curves in groups.items():
            for metric, values in curves.items():
                arrays[f"{noise_key}_{group_name}_{metric}"] = values
    return arrays


def save_noisy_probe_csv(
    path: Path,
    env_steps: np.ndarray,
    results: dict[float, dict[str, dict[str, np.ndarray]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["noise_std", "source", "model_step", "env_step", "metric", "value"],
        )
        writer.writeheader()
        for noise_std, groups in results.items():
            for source, curves in groups.items():
                for metric, values in sorted(curves.items()):
                    for i, value in enumerate(values):
                        writer.writerow(
                            {
                                "noise_std": float(noise_std),
                                "source": source,
                                "model_step": int(i + 1),
                                "env_step": int(env_steps[i]),
                                "metric": metric,
                                "value": float(value),
                            }
                        )


def plot_noise_metric(
    path: Path,
    env_steps: np.ndarray,
    results: dict[float, dict[str, dict[str, np.ndarray]]],
    metric: str,
    title: str,
    ylabel: str,
    *,
    include_encoded_gt: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for noise_std, groups in sorted(results.items()):
        if metric in groups["imagined"]:
            ax.plot(env_steps, groups["imagined"][metric], linewidth=2, label=f"imagined std={noise_std:g}")
        if include_encoded_gt and metric in groups["encoded_gt"]:
            ax.plot(
                env_steps,
                groups["encoded_gt"][metric],
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
                label=f"encoded sim std={noise_std:g}",
            )
    ax.set_title(title)
    ax.set_xlabel("Environment steps after context")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_noisy_probe_plots(
    output_dir: Path,
    env_steps: np.ndarray,
    results: dict[float, dict[str, dict[str, np.ndarray]]],
) -> None:
    specs = [
        ("agent_pos_rmse_px", "Agent Position RMSE by Noise", "RMSE (px)", None),
        ("block_pos_rmse_px", "Block Position RMSE by Noise", "RMSE (px)", None),
        ("block_angle_rmse_deg", "Block Angle RMSE by Noise", "RMSE (deg)", None),
        ("block_rel_objective_xy_rmse_px", "T Relative to Objective XY RMSE by Noise", "RMSE (px)", None),
        ("block_rel_objective_angle_rmse_deg", "T Relative to Objective Angle RMSE by Noise", "RMSE (deg)", None),
        ("block_rel_agent_xy_rmse_px", "T Relative to Agent XY RMSE by Noise", "RMSE (px)", None),
        ("block_rel_agent_angle_rmse_deg", "T Relative to Agent Angle RMSE by Noise", "RMSE (deg)", None),
        ("objective_met_mean_probability", "Mean P(Objective Met) by Noise", "Probability", None),
        ("objective_met_false_positive_rate", "Objective-Met FPR by Noise", "Rate", (0.0, 1.05)),
        ("objective_met_recall", "Objective-Met Recall by Noise", "Rate", (0.0, 1.05)),
        ("objective_met_precision", "Objective-Met Precision by Noise", "Rate", (0.0, 1.05)),
        ("objective_met_positive_count", "Objective-Met Support by Noise", "Count", None),
        ("latent_rmse", "Imagined vs Encoded Sim Latent RMSE by Noise", "Latent RMSE", None),
        ("latent_cosine", "Imagined vs Encoded Sim Latent Cosine by Noise", "Mean cosine", (-1.0, 1.0)),
    ]
    for metric, title, ylabel, ylim in specs:
        plot_noise_metric(
            output_dir / f"{metric}_by_noise.png",
            env_steps,
            results,
            metric,
            title,
            ylabel,
            include_encoded_gt=not metric.startswith("latent_"),
            ylim=ylim,
        )


def evaluate_one_noise(
    args: argparse.Namespace,
    *,
    h5: h5py.File,
    starts: list[tuple[int, int]],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    model: torch.nn.Module,
    history_size: int,
    probes: dict[str, object],
    classifier_probes: dict[str, object],
    noise_std: float,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    pred_embs = []
    gt_embs = []
    sim_states = []
    rng = np.random.default_rng(args.seed + 10_000 + int(round(noise_std * 1_000_000)))
    noise_args = argparse.Namespace(**vars(args))
    noise_args.action_noise_std = noise_std

    for batch_start in range(0, len(starts), args.batch_size):
        batch_starts = starts[batch_start : batch_start + args.batch_size]
        context_pixels, _, init_states = load_context_future_state(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=args.horizon,
            frameskip=args.frameskip,
        )
        action_blocks, noisy_actions = build_noisy_action_blocks(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=args.horizon,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
            normalize_actions=not args.no_normalize_actions,
            action_low=action_low,
            action_high=action_high,
            args=noise_args,
            rng=rng,
        )
        sim_pixels, states = rollout_simulator(
            init_states,
            noisy_actions,
            horizon=args.horizon,
            frameskip=args.frameskip,
        )
        pred_emb = rollout_embeddings(
            model,
            context_pixels,
            action_blocks,
            horizon=args.horizon,
            history_size=history_size,
            device=device,
        ).detach().cpu().float().numpy()
        gt_emb = probe_eval.encode_future_embeddings(
            model,
            sim_pixels,
            encode_batch_size=args.encode_batch_size,
            device=device,
        )
        pred_embs.append(pred_emb)
        gt_embs.append(gt_emb)
        sim_states.append(states)
        print(
            f"noise std {noise_std:g}: processed "
            f"{min(batch_start + len(batch_starts), len(starts))}/{len(starts)} trajectories"
        )

    pred_emb = np.concatenate(pred_embs, axis=0)
    gt_emb = np.concatenate(gt_embs, axis=0)
    states = np.concatenate(sim_states, axis=0)

    imagined_pred = probe_eval.probe_predictions(probes, pred_emb)
    encoded_gt_pred = probe_eval.probe_predictions(probes, gt_emb)
    imagined_class_pred = probe_eval.classifier_predictions(classifier_probes, pred_emb)
    encoded_gt_class_pred = probe_eval.classifier_predictions(classifier_probes, gt_emb)

    imagined_curves = probe_eval.compute_curves(imagined_pred, states, args)
    encoded_gt_curves = probe_eval.compute_curves(encoded_gt_pred, states, args)
    objective_threshold = classifier_probes["objective_met"].threshold
    imagined_curves.update(
        probe_eval.binary_curve_metrics(
            imagined_class_pred["objective_met"],
            states,
            args,
            threshold=objective_threshold,
        )
    )
    encoded_gt_curves.update(
        probe_eval.binary_curve_metrics(
            encoded_gt_class_pred["objective_met"],
            states,
            args,
            threshold=objective_threshold,
        )
    )
    imagined_curves["latent_rmse"] = probe_eval.latent_curve(pred_emb, gt_emb)
    imagined_curves["latent_cosine"] = probe_eval.latent_cosine_curve(pred_emb, gt_emb)
    return {"imagined": imagined_curves, "encoded_gt": encoded_gt_curves}


def run_evaluate_rollouts(args: argparse.Namespace) -> None:
    output_dir = repo_path(
        args.output_dir or f"models/rollout_probe/{Path(args.probe_dir).name}_{args.probe_kind}_noisy_actions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset_path = repo_path(args.dataset_path)
    checkpoint_cache_dir = repo_path(args.checkpoint_cache_dir)
    probe_dir = repo_path(args.probe_dir)
    noise_stds = parse_noise_stds(args.noise_stds)

    probes, classifier_probes = probe_eval.load_probes(probe_dir, args.probe_kind)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=checkpoint_cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", 3))
    env_steps = args.frameskip * np.arange(1, args.horizon + 1)

    probe_env = make_pusht_env(render_obs=False, relative=True, disable_env_checker=True)
    action_low = np.asarray(probe_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(probe_env.action_space.high, dtype=np.float32)
    probe_env.close()

    results = {}
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
        sampled = starts
        for noise_std in noise_stds:
            results[noise_std] = evaluate_one_noise(
                args,
                h5=h5,
                starts=starts,
                action_mean=action_mean,
                action_std=action_std,
                action_low=action_low,
                action_high=action_high,
                model=model,
                history_size=history_size,
                probes=probes,
                classifier_probes=classifier_probes,
                noise_std=noise_std,
                device=device,
            )

    flat_curves = flatten_curves_by_noise(results)
    save_noisy_probe_csv(output_dir / "noisy_probe_rollout_metrics.csv", env_steps, results)
    save_noisy_probe_plots(output_dir, env_steps, results)
    np.savez_compressed(
        output_dir / "noisy_probe_rollout_arrays.npz",
        env_steps=env_steps,
        noise_stds=np.asarray(noise_stds, dtype=np.float32),
        sampled=np.asarray(sampled, dtype=np.int64),
        **flat_curves,
    )

    final_step = {
        f"std_{noise_std:g}_{source}_{metric}": float(curves[metric][-1])
        for noise_std, groups in results.items()
        for source, curves in groups.items()
        for metric in (
            "agent_pos_rmse_px",
            "block_pos_rmse_px",
            "block_angle_rmse_deg",
            "block_rel_objective_xy_rmse_px",
            "block_rel_agent_xy_rmse_px",
            "objective_met_false_positive_rate",
            "objective_met_recall",
            "objective_met_precision",
            "objective_met_positive_count",
        )
        if metric in curves
    }
    summary = {
        "config": vars(args),
        "num_trajectories": len(sampled),
        "history_size": history_size,
        "noise_stds": noise_stds,
        "objective_met_threshold": classifier_probes["objective_met"].threshold,
        "env_steps": env_steps.tolist(),
        "action_noise": {
            "mode": args.action_noise_mode,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
        },
        "final_step": final_step,
    }
    with (output_dir / "noisy_probe_rollout_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(final_step, indent=2))
    print(f"saved noisy probe evaluation outputs to {output_dir}")


def run_visualize(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir or "models/rollout_decode/pusht_lewm_noisy_actions")
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
    rng = np.random.default_rng(args.seed + 10_000)

    sampled: list[tuple[int, int]] = []
    noisy_action_records: list[np.ndarray] = []
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

        probe_env = make_pusht_env(render_obs=False, relative=True, disable_env_checker=True)
        action_low = np.asarray(probe_env.action_space.low, dtype=np.float32)
        action_high = np.asarray(probe_env.action_space.high, dtype=np.float32)
        probe_env.close()

        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start : batch_start + args.batch_size]
            context_pixels, expert_future_pixels, init_states = load_context_future_state(
                h5,
                batch_starts,
                context_steps=args.context_steps,
                horizon=args.horizon,
                frameskip=args.frameskip,
            )
            action_blocks, noisy_actions = build_noisy_action_blocks(
                h5,
                batch_starts,
                context_steps=args.context_steps,
                horizon=args.horizon,
                frameskip=args.frameskip,
                action_mean=action_mean,
                action_std=action_std,
                normalize_actions=not args.no_normalize_actions,
                action_low=action_low,
                action_high=action_high,
                args=args,
                rng=rng,
            )

            noisy_pixels = rollout_simulator_images(
                init_states,
                noisy_actions,
                horizon=args.horizon,
                frameskip=args.frameskip,
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
            noisy_real = pixels_to_tensor(noisy_pixels)
            expert_real = pixels_to_tensor(expert_future_pixels)

            metric_accumulator.add(
                image_metric_parts(decoded, noisy_real, "imagined_vs_noisy_real", args)
            )
            metric_accumulator.add(
                image_metric_parts(noisy_real, expert_real, "noisy_real_vs_expert", args)
            )
            metric_accumulator.add(
                image_metric_parts(decoded, expert_real, "imagined_vs_expert", args)
            )

            for i, (ep, start) in enumerate(batch_starts):
                traj_name = f"trajectory_{len(sampled):03d}_ep{ep}_start{start}"
                suffix = ".gif" if args.video_format == "gif" else ".mp4"
                if not args.no_videos:
                    save_video(
                        output_dir / f"{traj_name}_side_by_side{suffix}",
                        noisy_real[i],
                        decoded[i],
                        env_steps,
                        args.fps,
                    )
                if not args.no_grids:
                    save_timestep_grid(
                        output_dir / f"{traj_name}_grid.png",
                        noisy_real[i],
                        decoded[i],
                        grid_env_steps,
                        args.frameskip,
                    )
                sampled.append((ep, start))
                noisy_action_records.append(noisy_actions[i].copy())
                if args.no_videos and args.no_grids:
                    print(f"processed noisy-action metrics for {traj_name}")
                else:
                    print(f"saved noisy-action visuals for {traj_name}")

    metric_curves = metric_accumulator.curves()
    save_metrics_csv(output_dir / "noisy_action_rollout_metrics.csv", env_steps, metric_curves)
    save_noisy_metric_plots(output_dir, env_steps, metric_curves)
    save_noisy_action_csv(output_dir / "noisy_actions.csv", sampled, noisy_action_records)
    np.savez_compressed(
        output_dir / "noisy_action_rollout_metric_arrays.npz",
        env_steps=env_steps,
        **metric_curves,
    )

    summary = {
        "config": vars(args),
        "num_trajectories": len(sampled),
        "sampled": [{"episode": int(ep), "start": int(start)} for ep, start in sampled],
        "history_size": history_size,
        "env_steps": env_steps.tolist(),
        "grid_env_steps": grid_env_steps.tolist(),
        "final_step": {key: float(value[-1]) for key, value in metric_curves.items()},
        "mean_over_horizon": {key: float(np.nanmean(value)) for key, value in metric_curves.items()},
        "action_noise": {
            "mode": args.action_noise_mode,
            "std_or_half_width": args.action_noise_std,
            "seed": args.seed + 10_000,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
        },
        "action_normalization": {
            "enabled": not args.no_normalize_actions,
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
    }
    with (output_dir / "noisy_action_rollout_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "evaluate_rollouts":
        run_evaluate_rollouts(args)
    else:
        run_visualize(args)


if __name__ == "__main__":
    main()
