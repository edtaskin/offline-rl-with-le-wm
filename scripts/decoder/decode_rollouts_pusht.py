"""Visualize PushT LeWM imagined rollouts with a trained latent image decoder.

This is the image-space companion to rollout_probe_pusht.py. It initializes LeWM
from ground-truth context frames, rolls forward with ground-truth action blocks,
decodes each imagined latent to an RGB frame, and saves:

  * side-by-side videos of real future frames vs decoded imagined frames,
  * a grid at model steps 1..10, i.e. environment steps 5..50 by default.

Relative paths are resolved from the top-level wrapper repo, regardless of the
current working directory used to launch the script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
import stable_worldmodel as swm
import torch
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
            real = pixels_to_tensor(future_pixels)
            env_steps = args.frameskip * np.arange(1, args.horizon + 1)

            for i, (ep, start) in enumerate(batch_starts):
                traj_name = f"trajectory_{len(sampled):03d}_ep{ep}_start{start}"
                suffix = ".gif" if args.video_format == "gif" else ".mp4"
                save_video(
                    output_dir / f"{traj_name}_side_by_side{suffix}",
                    real[i],
                    decoded[i],
                    env_steps,
                    args.fps,
                )
                save_timestep_grid(
                    output_dir / f"{traj_name}_grid.png",
                    real[i],
                    decoded[i],
                    grid_env_steps,
                    args.frameskip,
                )
                sampled.append((ep, start))
                print(f"saved visuals for {traj_name}")

    summary = {
        "config": vars(args),
        "num_trajectories": len(sampled),
        "sampled": [{"episode": int(ep), "start": int(start)} for ep, start in sampled],
        "history_size": history_size,
        "env_steps": (args.frameskip * np.arange(1, args.horizon + 1)).tolist(),
        "grid_env_steps": grid_env_steps.tolist(),
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
