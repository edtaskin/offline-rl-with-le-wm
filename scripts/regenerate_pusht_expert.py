#!/usr/bin/env python3
"""Re-render the PushT expert dataset at LeWM's native 224x224 resolution.

The downloaded diffusion-policy dataset contains 96x96 float images, while
LeWM consumes 224x224 images. Resizing those stored pixels cannot recover the
lost edges and detail. The default ``render`` mode instead restores every
stored simulator state in this repository's PushT environment and renders it
directly at the requested resolution.

The expert demonstrations use the fixed green-T goal at
``[256, 256, pi/4]``. Re-rendering uses that goal without the evaluation-time
alignment wrapper, because alignment would transform the recorded states.

The output image array is backed by a temporary on-disk memmap while it is
built. This keeps generation memory bounded even though the full 224x224
dataset is several gigabytes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import PUSHT_FIXED_TARGET_POSE, make_pusht_env  # noqa: E402


DEFAULT_DATASET = Path("data/expert_trajectories/pusht_expert.npz")
DEFAULT_OUTPUT_DATASET = Path("data/expert_trajectories/pusht_expert_224.npz")
DEFAULT_RESOLUTION = 224


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dataset",
        "--output_dataset",
        dest="output_dataset",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Square output resolution (default: LeWM's 224).",
    )
    parser.add_argument(
        "--mode",
        choices=("render", "upsample"),
        default="render",
        help="render=re-draw from states; upsample=bilinear-resize stored pixels.",
    )
    parser.add_argument(
        "--goal-pose",
        "--goal_pose",
        dest="goal_pose",
        type=float,
        nargs=3,
        default=PUSHT_FIXED_TARGET_POSE.tolist(),
        metavar=("X", "Y", "ANGLE"),
        help="Fixed goal pose used by the demonstrations.",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=0,
        help="Compare the first N renders, downscaled, with the stored frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional positive frame cap for a smoke test.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Compress the output NPZ (smaller but substantially slower to save/load).",
    )
    return parser.parse_args(argv)


def _read_npy_header(stream):
    version = np.lib.format.read_magic(stream)
    shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def npz_array_metadata(path: Path, key: str):
    """Read an NPZ member's shape/dtype without materializing the array."""
    with zipfile.ZipFile(path) as archive:
        member = f"{key}.npy"
        if member not in archive.namelist():
            raise KeyError(f"{path} missing required key '{key}'")
        with archive.open(member) as stream:
            return _read_npy_header(stream)


def iter_npz_image_batches(path: Path, *, count: int, batch_size: int = 64):
    """Yield leading NHWC image batches directly from an NPZ member."""
    with zipfile.ZipFile(path) as archive, archive.open("images.npy") as stream:
        shape, fortran_order, dtype = _read_npy_header(stream)
        if fortran_order:
            raise ValueError("Fortran-ordered image arrays are not supported")
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"expected NHWC RGB images, got shape {shape}")
        if not 0 <= count <= shape[0]:
            raise ValueError(f"requested {count} images from an array of length {shape[0]}")

        frame_shape = shape[1:]
        values_per_frame = int(np.prod(frame_shape))
        remaining = count
        while remaining:
            current = min(batch_size, remaining)
            byte_count = current * values_per_frame * dtype.itemsize
            payload = stream.read(byte_count)
            if len(payload) != byte_count:
                raise ValueError(f"truncated images array in {path}")
            yield np.frombuffer(payload, dtype=dtype).reshape((current, *frame_shape))
            remaining -= current


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32, copy=False)
    if image.size and float(image.max()) <= 1.5:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def upsample_images(
    dataset_path: Path,
    output: np.ndarray,
    *,
    count: int,
    resolution: int,
) -> None:
    import cv2

    offset = 0
    progress = tqdm(total=count, desc="upsample")
    for batch in iter_npz_image_batches(dataset_path, count=count):
        for image in batch:
            output[offset] = cv2.resize(
                to_uint8_image(image),
                (resolution, resolution),
                interpolation=cv2.INTER_LINEAR,
            )
            offset += 1
        progress.update(len(batch))
    progress.close()


def make_goal_state(goal_pose: np.ndarray, state_dim: int) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(3)
    if state_dim not in (5, 7):
        raise ValueError(f"expected 5D or 7D PushT states, got {state_dim}D")
    # Agent pose is irrelevant to the green goal rendering; use the workspace center.
    state = np.array([256.0, 256.0, *goal_pose], dtype=np.float64)
    if state_dim == 7:
        state = np.concatenate((state, np.zeros(2, dtype=np.float64)))
    return state


def render_images_from_states(
    states: np.ndarray,
    output: np.ndarray,
    *,
    resolution: int,
    goal_pose: np.ndarray,
) -> None:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(3)
    goal_state = make_goal_state(goal_pose, states.shape[1])

    # Use the bare environment. The fixed-target alignment wrapper is correct for
    # sampled evaluation tasks, but would incorrectly warp recorded expert states.
    env = make_pusht_env(
        render_mode="rgb_array",
        render_obs=False,
        resolution=resolution,
        relative=False,
        sync_goal_pose=False,
        align_sampled_goal_to_fixed_target=False,
        max_episode_steps=int(len(states) + 1),
    )
    env.reset(options={"state": states[0], "goal_state": goal_state})
    unwrapped = env.unwrapped
    unwrapped.goal_pose = goal_pose.copy()
    unwrapped.goal_state = goal_state.copy()

    try:
        for index, state in enumerate(tqdm(states, desc="render")):
            unwrapped._set_state(np.asarray(state, dtype=np.float64))
            # Keep the goal fixed even if a future environment version samples it.
            unwrapped.goal_pose = goal_pose
            output[index] = np.asarray(unwrapped.render(), dtype=np.uint8)
    finally:
        env.close()


def verify_against_original(
    rendered: np.ndarray,
    originals: np.ndarray,
) -> float:
    import cv2

    if not len(originals):
        raise ValueError("verification requires at least one frame")
    maes = []
    for rendered_frame, original_frame in zip(rendered, originals):
        expected = to_uint8_image(original_frame)
        actual = rendered_frame
        if actual.shape[:2] != expected.shape[:2]:
            actual = cv2.resize(
                actual,
                (expected.shape[1], expected.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        maes.append(
            float(np.mean(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        )
    return float(np.mean(maes))


def _load_image_prefix(path: Path, count: int) -> np.ndarray:
    batches = list(iter_npz_image_batches(path, count=count, batch_size=max(count, 1)))
    return np.concatenate(batches, axis=0)


def regenerate_dataset(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output_dataset)
    if dataset_path.suffix != ".npz" or output_path.suffix != ".npz":
        raise ValueError("dataset and output-dataset must use the .npz extension")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.verify_n < 0:
        raise ValueError("verify-n must be non-negative")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be at least 1")
    if dataset_path.resolve() == output_path.resolve():
        raise ValueError("output-dataset must differ from the input dataset")

    image_shape, _, _ = npz_array_metadata(dataset_path, "images")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        for key in ("states", "actions", "episode_ends"):
            if key not in dataset:
                raise KeyError(f"{dataset_path} missing required key '{key}'")
        states = np.asarray(dataset["states"])
        actions = np.asarray(dataset["actions"])
        episode_ends = np.asarray(dataset["episode_ends"])

    total_frames = len(states)
    if len(image_shape) != 4 or image_shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB images, got shape {image_shape}")
    if total_frames != image_shape[0] or total_frames != len(actions):
        raise ValueError(
            f"length mismatch: states={total_frames} actions={len(actions)} "
            f"images={image_shape[0]}"
        )
    if states.ndim != 2 or states.shape[1] not in (5, 7):
        raise ValueError(f"expected states shaped (N, 5) or (N, 7), got {states.shape}")
    if not len(episode_ends) or int(episode_ends[-1]) != total_frames:
        raise ValueError("episode_ends must be non-empty and end at the dataset length")

    frame_count = (
        total_frames if args.max_frames is None else min(args.max_frames, total_frames)
    )
    states = states[:frame_count]
    actions = actions[:frame_count]
    episode_ends = episode_ends[episode_ends <= frame_count]
    if not len(episode_ends) or int(episode_ends[-1]) != frame_count:
        episode_ends = np.concatenate(
            (episode_ends, np.asarray([frame_count], dtype=episode_ends.dtype))
        )

    goal_pose = np.asarray(args.goal_pose, dtype=np.float64)
    print(
        f"Loaded {dataset_path} | frames={frame_count}/{total_frames} "
        f"episodes={len(episode_ends)} mode={args.mode} "
        f"resolution={args.resolution}x{args.resolution} "
        f"goal_pose={goal_pose.tolist()}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".images.npy",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    archive_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".npz",
        dir=output_path.parent,
        delete=False,
    )
    archive_path = Path(archive_file.name)
    archive_file.close()
    images_out = None
    try:
        images_out = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, args.resolution, args.resolution, 3),
        )
        if args.mode == "upsample":
            upsample_images(
                dataset_path,
                images_out,
                count=frame_count,
                resolution=args.resolution,
            )
        else:
            render_images_from_states(
                states,
                images_out,
                resolution=args.resolution,
                goal_pose=goal_pose,
            )
        images_out.flush()

        if args.verify_n:
            verify_count = min(args.verify_n, frame_count)
            originals = _load_image_prefix(dataset_path, verify_count)
            mae = verify_against_original(images_out[:verify_count], originals)
            print(
                f"verify: compared {verify_count} frames | mean pixel MAE vs original "
                f"(after downscale if needed) = {mae:.4f}"
            )

        save = np.savez_compressed if args.compressed else np.savez
        save(
            archive_path,
            states=states,
            actions=actions,
            images=images_out,
            episode_ends=episode_ends,
        )
        archive_path.replace(output_path)
    finally:
        if images_out is not None:
            del images_out
        temp_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)

    print(
        f"Saved {output_path} | images="
        f"({frame_count}, {args.resolution}, {args.resolution}, 3) dtype=uint8"
    )
    return output_path


def main(argv=None) -> None:
    regenerate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
