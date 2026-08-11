"""Generate a PushT non-expert dataset from expert actions plus noise.

The dataset stores simulator frames only at world-model steps, not every env
step. This is intended for probing whether LeWM state probes remain reliable on
states reached by noisy, partially off-policy trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rollouts.noisy_actions import (  # noqa: E402
    build_noisy_action_blocks,
    load_context_future_state,
    rollout_simulator,
)
from scripts.decoder.gt_rollout_video import sample_starts  # noqa: E402
from src.envs import make_pusht_env  # noqa: E402


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_noise_stds(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("--noise-stds must contain at least one value")
    if any(v < 0 for v in values):
        raise ValueError("--noise-stds must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dataset", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--output", default="models/probes/pusht_noisy_actions/noisy_action_dataset.h5")
    parser.add_argument("--num-trajectories", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--noise-stds", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--action-noise-mode", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_indices(num_items: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_items).astype(np.int64)
    if num_items >= 3:
        n_val = max(1, int(0.1 * num_items))
        n_test = max(1, int(0.1 * num_items))
        n_train = max(1, num_items - n_val - n_test)
    else:
        n_train = num_items
        n_val = 0
        n_test = 0
    return {
        "train": np.sort(idx[:n_train]),
        "val": np.sort(idx[n_train : n_train + n_val]),
        "test": np.sort(idx[n_train + n_val : n_train + n_val + n_test]),
    }


def h5_compression_kwargs(args: argparse.Namespace) -> dict:
    if not args.compression or args.compression.lower() == "none":
        return {}
    kwargs = {"compression": args.compression}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.compression_level)
    return kwargs


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:].astype(np.float32)
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def main() -> None:
    args = parse_args()
    output = repo_path(args.output)
    if output.exists() and not args.force:
        print(f"dataset exists, keeping: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    noise_stds = parse_noise_stds(args.noise_stds)
    total_trajectories = args.num_trajectories * len(noise_stds)
    total_frames = total_trajectories * args.horizon
    compression_kwargs = h5_compression_kwargs(args)

    with h5py.File(repo_path(args.expert_dataset), "r") as src:
        starts = sample_starts(src, args.num_trajectories, args.context_steps, args.horizon, args.frameskip, args.seed)
        action_mean, action_std = action_stats(src)
        probe_env = make_pusht_env(disable_env_checker=True)
        try:
            action_low = np.asarray(probe_env.action_space.low, dtype=np.float32)
            action_high = np.asarray(probe_env.action_space.high, dtype=np.float32)
        finally:
            probe_env.close()

        tmp = output.with_suffix(output.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with h5py.File(tmp, "w") as dst:
            pixels_ds = dst.create_dataset(
                "pixels",
                shape=(total_frames, args.resolution, args.resolution, 3),
                dtype=np.uint8,
                chunks=(1, args.resolution, args.resolution, 3),
                **compression_kwargs,
            )
            state_ds = dst.create_dataset("state", shape=(total_frames, src["state"].shape[1]), dtype=np.float32)
            action_ds = dst.create_dataset("action", shape=(total_frames, args.frameskip, 2), dtype=np.float32)
            noise_ds = dst.create_dataset("noise_std", shape=(total_frames,), dtype=np.float32)
            source_episode_ds = dst.create_dataset("source_episode", shape=(total_frames,), dtype=np.int64)
            source_start_ds = dst.create_dataset("source_start", shape=(total_frames,), dtype=np.int64)
            model_step_ds = dst.create_dataset("model_step", shape=(total_frames,), dtype=np.int64)

            write_at = 0
            for noise_idx, noise_std in enumerate(noise_stds):
                local_args = argparse.Namespace(
                    action_noise_std=float(noise_std),
                    action_noise_mode=args.action_noise_mode,
                )
                rng = np.random.default_rng(args.seed + 10_000 + noise_idx)
                for start_idx in range(0, len(starts), args.batch_size):
                    batch_starts = starts[start_idx : start_idx + args.batch_size]
                    _, _, init_states = load_context_future_state(
                        src,
                        batch_starts,
                        context_steps=args.context_steps,
                        horizon=args.horizon,
                        frameskip=args.frameskip,
                    )
                    _, noisy_actions = build_noisy_action_blocks(
                        src,
                        batch_starts,
                        context_steps=args.context_steps,
                        horizon=args.horizon,
                        frameskip=args.frameskip,
                        action_mean=action_mean,
                        action_std=action_std,
                        normalize_actions=False,
                        action_low=action_low,
                        action_high=action_high,
                        args=local_args,
                        rng=rng,
                    )
                    pixels, states = rollout_simulator(
                        init_states,
                        noisy_actions,
                        horizon=args.horizon,
                        frameskip=args.frameskip,
                        resolution=args.resolution,
                    )
                    n = len(batch_starts) * args.horizon
                    sl = slice(write_at, write_at + n)
                    pixels_ds[sl] = pixels.reshape(n, args.resolution, args.resolution, 3)
                    state_ds[sl] = states.reshape(n, states.shape[-1]).astype(np.float32)
                    action_ds[sl] = noisy_actions.reshape(len(batch_starts), args.horizon, args.frameskip, 2).reshape(
                        n, args.frameskip, 2
                    )
                    noise_ds[sl] = float(noise_std)
                    source_episode_ds[sl] = np.repeat([ep for ep, _ in batch_starts], args.horizon)
                    source_start_ds[sl] = np.repeat([start for _, start in batch_starts], args.horizon)
                    model_step_ds[sl] = np.tile(np.arange(1, args.horizon + 1, dtype=np.int64), len(batch_starts))
                    write_at += n
                    print(
                        f"noise {noise_std:g}: "
                        f"{min(start_idx + len(batch_starts), len(starts))}/{len(starts)} trajectories"
                    )

            splits = split_indices(total_frames, args.seed + 99)
            for split, idx in splits.items():
                dst.create_dataset(f"{split}_idx", data=idx, dtype=np.int64)

            metadata = {
                "expert_dataset": str(repo_path(args.expert_dataset)),
                "num_base_trajectories": int(args.num_trajectories),
                "num_trajectories": int(total_trajectories),
                "horizon": int(args.horizon),
                "context_steps": int(args.context_steps),
                "frameskip": int(args.frameskip),
                "noise_stds": noise_stds,
                "action_noise_mode": args.action_noise_mode,
                "seed": int(args.seed),
                "resolution": int(args.resolution),
                "action_low": action_low.tolist(),
                "action_high": action_high.tolist(),
                "stored_frames": int(total_frames),
                "splits": {split: int(len(idx)) for split, idx in splits.items()},
            }
            dst.attrs["metadata"] = json.dumps(metadata)
            dst.create_dataset("metadata_json", data=np.bytes_(json.dumps(metadata)))
        tmp.replace(output)
    print(f"saved noisy-action dataset: {output}")


if __name__ == "__main__":
    main()
