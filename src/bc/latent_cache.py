from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.bc.lewm import lewm_preprocessing_metadata


def path_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def default_latent_cache_path(data_path):
    data_path = Path(data_path)
    cache_name = f"{data_path.stem}_lewm_object_imagenet224_cls.pt"
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
):
    preprocessing = lewm_preprocessing_metadata(normalization)
    return {
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
