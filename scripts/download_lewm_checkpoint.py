"""Download and convert the pretrained LeWM PushT encoder checkpoint.

The BC / latent-PPO pipelines load a serialized LeWM *object* checkpoint via
``torch.load(..., weights_only=False).encoder``. The official Hugging Face mirror
(``quentinll/lewm-pusht``) instead ships a ``weights.pt`` state dict plus a
``config.json``, so this script performs the one-time conversion:

1. download ``weights.pt`` + ``config.json`` into ``<swm cache>/hf_pusht``;
2. rebuild the model from ``config.json`` -- a Hydra config whose top-level
   ``_target_`` is ``stable_worldmodel.wm.lewm.LeWM`` with nested ``_target_``
   entries -- via ``hydra.utils.instantiate``;
3. load ``weights.pt`` and ``torch.save`` the object to the path the loader
   expects (``<swm cache>/checkpoints/pusht/lewm_object.ckpt``; see
   :func:`src.ppo.lewm_encoder.default_lewm_checkpoint_path`).

Usage::

    python -m scripts.download_lewm_checkpoint          # download + convert (idempotent)
    python -m scripts.download_lewm_checkpoint --force   # redo even if it exists
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The HF `weights.pt` was saved with an older `transformers` whose ViTModel used
# the classic module names; newer transformers (>=5) renamed them. This 1:1 map
# (same tensor shapes, identical math) realigns an old-style encoder state dict
# to the current naming. Applied only if a plain strict load fails.
_VIT_KEY_RENAMES = (
    ("encoder.encoder.layer.", "encoder.layers."),
    (".attention.attention.query", ".attention.q_proj"),
    (".attention.attention.key", ".attention.k_proj"),
    (".attention.attention.value", ".attention.v_proj"),
    (".attention.output.dense", ".attention.o_proj"),  # must precede .output.dense
    (".intermediate.dense", ".mlp.fc1"),
    (".output.dense", ".mlp.fc2"),
)


def _remap_vit_encoder_keys(state_dict: dict) -> dict:
    """Rename old-style HF ViT encoder keys to the current transformers layout."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.encoder.layer."):
            for old, new in _VIT_KEY_RENAMES:
                new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def _load_lewm_state_dict(model, state_dict) -> None:
    """Load weights, realigning old-style ViT encoder keys if needed (strict)."""
    import torch

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass  # naming mismatch -> try the encoder key remap below

    remapped = _remap_vit_encoder_keys(state_dict)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "LeWM weights do not fit the instantiated model even after encoder key "
            f"remap.\n  missing (first 5): {list(missing)[:5]}\n"
            f"  unexpected (first 5): {list(unexpected)[:5]}"
        )
    print("Realigned old-style HF ViT encoder keys to the current transformers layout.")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001 - dotenv is optional
    pass


def download_and_convert(repo_id: str, force: bool) -> Path:
    import hydra.utils
    import torch
    import stable_worldmodel as swm

    from src.ppo.lewm_encoder import default_lewm_checkpoint_path

    out_path = default_lewm_checkpoint_path()
    if out_path.exists() and not force:
        print(f"Object checkpoint already present at {out_path} (use --force to redo).")
        return out_path

    cache_dir = Path(swm.data.utils.get_cache_dir())
    hf_dir = cache_dir / "hf_pusht"

    # 1) download weights.pt + config.json from the HF mirror.
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    print(f"Downloading {repo_id} -> {hf_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(hf_dir),
        allow_patterns=["weights.pt", "config.json"],
        token=token,
    )

    # 2) rebuild the LeWM model from its Hydra config (recursive _target_).
    cfg = json.loads((hf_dir / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)

    # 3) load weights and save the object checkpoint where the loader expects it.
    state_dict = torch.load(hf_dir / "weights.pt", map_location="cpu", weights_only=False)
    _load_lewm_state_dict(model, state_dict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out_path)
    print(f"Saved LeWM object checkpoint to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + convert the LeWM PushT checkpoint")
    parser.add_argument("--repo-id", default="quentinll/lewm-pusht", help="HF model repo id")
    parser.add_argument("--force", action="store_true", help="reconvert even if the object ckpt exists")
    args = parser.parse_args()
    download_and_convert(args.repo_id, args.force)


if __name__ == "__main__":
    main()
