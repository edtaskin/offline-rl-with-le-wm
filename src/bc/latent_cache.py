from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.representations.lewm import (
    LEWM_LATENT_PROJECTED,
    LEWM_LATENT_RAW_CLS,
    LEWM_LATENT_REPRESENTATIONS,
    lewm_preprocessing_metadata,
)


def path_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def default_latent_cache_path(data_path, latent_representation=LEWM_LATENT_RAW_CLS):
    if latent_representation not in LEWM_LATENT_REPRESENTATIONS:
        raise ValueError(f"unsupported LeWM latent representation: {latent_representation!r}")
    data_path = Path(data_path)
    suffix = "cls" if latent_representation == LEWM_LATENT_RAW_CLS else "projected"
    cache_name = f"{data_path.stem}_lewm_object_imagenet224_{suffix}.pt"
    return str(Path("data", "latent_cache", cache_name))


def num_samples_from_data(data_path):
    with np.load(data_path, allow_pickle=True) as data:
        return int(data["actions"].shape[0])


def expected_latent_cache_metadata(
    data_path,
    checkpoint_path,
    num_samples,
    latent_dim,
    normalization="imagenet",
    latent_representation=LEWM_LATENT_RAW_CLS,
):
    if latent_representation not in LEWM_LATENT_REPRESENTATIONS:
        raise ValueError(f"unsupported LeWM latent representation: {latent_representation!r}")
    preprocessing = lewm_preprocessing_metadata(normalization)
    metadata = {
        "format_version": 1,
        "cache_type": "lewm_cls_latents",
        "data_file": path_fingerprint(data_path),
        "lewm_checkpoint_file": path_fingerprint(checkpoint_path),
        "num_samples": int(num_samples),
        "latent_dim": int(latent_dim),
        "image_size": list(preprocessing["image_size"]),
        "image_normalization": preprocessing["image_normalization"],
        "image_mean": list(preprocessing["image_mean"]),
        "image_std": list(preprocessing["image_std"]),
    }
    # Keep existing raw-CLS caches compatible. Projected caches must carry an
    # explicit contract because both representations happen to be 192-d.
    if latent_representation == LEWM_LATENT_PROJECTED:
        metadata.update(
            {
                "format_version": 2,
                "cache_type": "lewm_projected_latents",
                "latent_representation": LEWM_LATENT_PROJECTED,
            }
        )
    return metadata


def metadata_matches(actual, expected):
    if not isinstance(actual, dict):
        return False, ["metadata is missing or not a dict"]
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    return len(mismatches) == 0, mismatches


def check_latent_cache(cache_path, expected_metadata):
    if not cache_path or not Path(cache_path).exists():
        return False, ["cache file does not exist"]
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        return False, [f"cache could not be loaded: {exc}"]
    if not isinstance(payload, dict) or "latents" not in payload:
        return False, ["cache payload must be a dict containing a 'latents' tensor"]
    latents = payload["latents"]
    if not torch.is_tensor(latents):
        return False, ["cache 'latents' entry is not a tensor"]
    expected_shape = (
        int(expected_metadata["num_samples"]),
        int(expected_metadata["latent_dim"]),
    )
    if tuple(latents.shape) != expected_shape:
        return False, [f"latents shape mismatch: expected {expected_shape}, got {tuple(latents.shape)}"]
    return metadata_matches(payload.get("metadata"), expected_metadata)


def build_latent_cache(source_dataset, extractor, cache_path, metadata, batch_size):
    if batch_size < 1:
        raise ValueError("latent_cache_batch_size must be at least 1")
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building LeWM latent cache: {cache_path}")
    all_latents = []
    for start in range(0, len(source_dataset), batch_size):
        end = min(start + batch_size, len(source_dataset))
        latents = extractor.encode(source_dataset.images[start:end]).detach().cpu()
        all_latents.append(latents)
        if start == 0 or end == len(source_dataset):
            print(f"  encoded {end}/{len(source_dataset)} frames")
    cached_latents = torch.cat(all_latents, dim=0).contiguous()
    torch.save(
        {
            "latents": cached_latents,
            "metadata": {
                **metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        cache_path,
    )
    print(f"Saved LeWM latent cache with shape {tuple(cached_latents.shape)}")


def build_projected_latent_cache(
    raw_cache_path,
    projector,
    cache_path,
    metadata,
    batch_size,
):
    """Apply the frozen JEPA projector to a verified raw-CLS cache."""

    if batch_size < 1:
        raise ValueError("latent_cache_batch_size must be at least 1")
    payload = torch.load(raw_cache_path, map_location="cpu")
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("latents")):
        raise ValueError("raw latent cache must contain a 'latents' tensor")
    raw_latents = payload["latents"]
    try:
        device = next(projector.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    projector.eval()
    projected_batches = []
    print(f"Building projected LeWM latent cache from: {raw_cache_path}")
    with torch.inference_mode():
        for start in range(0, len(raw_latents), batch_size):
            end = min(start + batch_size, len(raw_latents))
            projected_batches.append(
                projector(raw_latents[start:end].to(device)).detach().cpu()
            )
    projected_latents = torch.cat(projected_batches, dim=0).contiguous()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": projected_latents,
            "metadata": {
                **metadata,
                "source_raw_cache": path_fingerprint(raw_cache_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        cache_path,
    )
    print(f"Saved projected LeWM latent cache with shape {tuple(projected_latents.shape)}")
