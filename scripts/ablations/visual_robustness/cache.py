"""Counterfactual PushT rendering and ablation-local latent caches."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.envs import make_pusht_env

from .encoders import encoder_metadata
from .shifts import VisualShiftSpec, get_condition


CACHE_FORMAT_VERSION = 1
DEFAULT_DATA_PATH = Path("data/expert_trajectories/pusht_expert.npz")
DEFAULT_OUTPUT_ROOT = Path("runs/ablations/lewm_visual_robustness")


def _sha256_array(array):
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def path_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_source_arrays(data_path=DEFAULT_DATA_PATH):
    with np.load(data_path, allow_pickle=False) as payload:
        arrays = {
            "states": np.asarray(payload["states"]).copy(),
            "actions": np.asarray(payload["actions"]).copy(),
            "episode_ends": np.asarray(payload["episode_ends"]).copy(),
        }
    count = len(arrays["states"])
    if len(arrays["actions"]) != count:
        raise ValueError("states/actions length mismatch")
    if not len(arrays["episode_ends"]) or int(arrays["episode_ends"][-1]) != count:
        raise ValueError("episode_ends must terminate at the source length")
    return arrays


def source_metadata(data_path, arrays):
    return {
        "file": path_fingerprint(data_path),
        "num_samples": int(len(arrays["states"])),
        "num_episodes": int(len(arrays["episode_ends"])),
        "states_sha256": _sha256_array(arrays["states"]),
        "actions_sha256": _sha256_array(arrays["actions"]),
        "episode_ends_sha256": _sha256_array(arrays["episode_ends"]),
    }


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def expected_cache_metadata(
    data_path,
    arrays,
    encoder_name,
    encoder,
    condition,
    resolution=96,
):
    resolution = condition.effective_resolution(resolution)
    metadata = encoder_metadata(encoder_name, encoder)
    metadata["checkpoint"] = path_fingerprint(metadata.pop("checkpoint_path"))
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_type": "visual_robustness_latents",
        "source": source_metadata(data_path, arrays),
        "renderer": {
            "environment": "swm/PushT-v1",
            "stable_worldmodel_version": _package_version("stable-worldmodel"),
            "resolution": int(resolution),
        },
        "condition": condition.to_dict(),
        "encoder": metadata,
    }


def cache_path(output_root, encoder_name, condition_name):
    return Path(output_root) / "cache" / encoder_name / f"{condition_name}.pt"


def metadata_matches(actual, expected):
    if not isinstance(actual, dict):
        return False, ["metadata is missing or is not a dictionary"]
    mismatches = []
    for key, value in expected.items():
        if actual.get(key) != value:
            mismatches.append(f"{key}: expected {value!r}, got {actual.get(key)!r}")
    return not mismatches, mismatches


def check_cache(path, expected):
    path = Path(path)
    if not path.exists():
        return False, ["cache file does not exist"]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, [f"cache cannot be loaded: {exc}"]
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("latents")):
        return False, ["cache must contain a latent tensor"]
    expected_shape = (
        expected["source"]["num_samples"],
        expected["encoder"]["feature_dim"],
    )
    if tuple(payload["latents"].shape) != expected_shape:
        return False, [
            f"latent shape mismatch: expected {expected_shape}, got {tuple(payload['latents'].shape)}"
        ]
    return metadata_matches(payload.get("metadata"), expected)


class CounterfactualRenderer:
    """Render saved physical states with one fixed visual intervention."""

    def __init__(self, condition: VisualShiftSpec, resolution=96):
        self.resolution = condition.effective_resolution(resolution)
        kwargs = {
            "render_obs": False,
            "resolution": self.resolution,
            # Downloaded demonstrations use PushT's fixed visual goal.
            "sync_goal_pose": False,
        }
        init_value = condition.renderer_init_value()
        if init_value is not None:
            kwargs["init_value"] = init_value
        self.env = make_pusht_env(**kwargs)
        self.env.reset(seed=0)
        self.condition = condition

    def render(self, state):
        unwrapped = self.env.unwrapped
        unwrapped.agent.velocity = (0.0, 0.0)
        unwrapped.block.velocity = (0.0, 0.0)
        unwrapped.block.angular_velocity = 0.0
        unwrapped._set_state(np.asarray(state, dtype=np.float64))
        frame = np.asarray(self.env.render(), dtype=np.uint8)
        return self.condition.apply_post_render(frame)

    def render_many(self, states):
        return np.stack([self.render(state) for state in states], axis=0)

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_cache(
    *,
    data_path,
    output_root,
    encoder_name,
    encoder,
    condition_name,
    batch_size=128,
    resolution=96,
    force=False,
):
    if batch_size < 1 or resolution < 1:
        raise ValueError("batch_size and resolution must be positive")
    arrays = load_source_arrays(data_path)
    condition = get_condition(condition_name)
    expected = expected_cache_metadata(
        data_path, arrays, encoder_name, encoder, condition, resolution
    )
    output_path = cache_path(output_root, encoder_name, condition_name)
    valid, messages = check_cache(output_path, expected)
    if valid and not force:
        print(f"Using valid cache: {output_path}")
        return output_path
    if output_path.exists() and not force:
        details = "\n  ".join(messages[:5])
        raise RuntimeError(
            f"incompatible cache exists at {output_path}; use --force to rebuild:\n  {details}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    states = arrays["states"]
    with CounterfactualRenderer(condition, resolution) as renderer:
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            frames = renderer.render_many(states[start:end])
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            parts.append(encoder.encode(images).detach().cpu())
            if start == 0 or end == len(states):
                print(f"  {encoder_name}/{condition_name}: encoded {end}/{len(states)}")
    latents = torch.cat(parts, dim=0).contiguous()
    torch.save(
        {
            "latents": latents,
            "metadata": {
                **expected,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        output_path,
    )
    return output_path


def save_manifest(output_root, data_path, condition_names):
    arrays = load_source_arrays(data_path)
    payload = {
        "source": source_metadata(data_path, arrays),
        "conditions": [get_condition(name).to_dict() for name in condition_names],
    }
    path = Path(output_root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def save_thumbnail_grid(
    *,
    data_path,
    output_root,
    condition_names,
    resolution=96,
    count=4,
):
    if count < 1 or resolution < 1:
        raise ValueError("thumbnail count and resolution must be positive")
    if not condition_names:
        raise ValueError("at least one visual condition is required")
    arrays = load_source_arrays(data_path)
    indices = np.linspace(0, len(arrays["states"]) - 1, count, dtype=int)
    rows = []
    label_width = 150
    for name in condition_names:
        condition = get_condition(name)
        with CounterfactualRenderer(condition, resolution) as renderer:
            frames = renderer.render_many(arrays["states"][indices])
        row = Image.new("RGB", (label_width + count * resolution, resolution), "white")
        ImageDraw.Draw(row).text((5, resolution // 2 - 6), name, fill="black")
        for column, frame in enumerate(frames):
            thumbnail = Image.fromarray(frame)
            if thumbnail.size != (resolution, resolution):
                thumbnail = thumbnail.resize((resolution, resolution), Image.Resampling.LANCZOS)
            row.paste(thumbnail, (label_width + column * resolution, 0))
        rows.append(row)
    grid = Image.new("RGB", (rows[0].width, len(rows) * resolution), "white")
    for index, row in enumerate(rows):
        grid.paste(row, (0, index * resolution))
    path = Path(output_root) / "visual_conditions.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return path
