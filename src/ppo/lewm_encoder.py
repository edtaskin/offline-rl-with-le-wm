"""Frozen LeWM (JEPA ViT) image encoder for latent PPO.

The BC pipeline encodes a PushT frame by running the official LeWM object
checkpoint's ViT encoder and taking its CLS token::

    latent = encoder(pixels).last_hidden_state[:, 0, :]   # [B, 192]

with the pixels preprocessed exactly as in training (``/255`` -> resize to
``224x224`` -> ImageNet normalization). See ``src/bc/train_bc_latent.py`` and
``src/bc/run_eval.py`` for the reference path.

:class:`LeWMLatentEncoder` wraps that raw encoder plus preprocessing behind a
plain ``forward(images) -> [B, latent_dim]`` interface, so the PPO actor/critic
in :mod:`src.ppo.agent` can call ``self.encoder(images)`` and get a
latent tensor back -- exactly like the ``DummyImageEncoder`` used for testing.
The encoder is frozen (``eval`` + ``requires_grad_(False)``): PPO fine-tunes the
BC policy and value head on top of fixed latents.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

# Reuse the exact preprocessing constants the dataset/eval use, so PPO latents
# match the ones the BC policy was trained on.
from src.bc.lewm import LEWM_IMAGE_MEAN, LEWM_IMAGE_SIZE, LEWM_IMAGE_STD

LEWM_LATENT_DIM = 192


def default_lewm_checkpoint_path() -> Path:
    """Path to the official LeWM object checkpoint in the swm cache.

    Mirrors ``train_bc_latent.py`` / ``run_eval.py``:
    ``<swm cache>/checkpoints/pusht/lewm_object.ckpt``.
    """
    swm = importlib.import_module("stable_worldmodel")
    return Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")


def _ensure_lewm_on_path() -> None:
    """Put the le-wm source dir on ``sys.path`` so the JEPA object can unpickle.

    The object checkpoint is a pickled ``jepa.JEPA`` instance, so ``torch.load``
    needs the le-wm ``jepa`` / ``module`` modules importable. Uses ``LE_WM_PATH``
    when set, otherwise the vendored ``le-wm/`` directory at the repo root.
    """
    env_path = os.getenv("LE_WM_PATH") or ""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(env_path) if env_path.strip() else repo_root / "le-wm"
    candidate = candidate.resolve()
    if (candidate / "jepa.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def load_lewm_encoder(
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """Load and freeze the raw LeWM ViT encoder from the object checkpoint.

    The conversion script stored the model via ``torch.save(model, out)``, so the
    checkpoint unpickles a live JEPA object; ``weights_only=False`` is required
    (PyTorch >= 2.6). We keep only ``model.encoder``.
    """
    if checkpoint_path is None:
        checkpoint_path = default_lewm_checkpoint_path()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"LeWM encoder checkpoint not found at {checkpoint_path}. "
            "Run `python -m scripts.download_lewm_checkpoint` to download + convert "
            "the official LeWM PushT checkpoint, or pass --encoder-checkpoint."
        )

    # The object checkpoint is a pickled JEPA instance; make its classes importable.
    _ensure_lewm_on_path()
    lewm_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = lewm_model.encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


class LeWMLatentEncoder(nn.Module):
    """Frozen LeWM encoder + preprocessing -> CLS latent ``[B, latent_dim]``.

    Accepts an image batch ``[B, C, H, W]`` that is either ``uint8`` in ``[0, 255]``
    or float in ``[0, 1]`` (values > 1.5 are treated as ``0..255`` and divided by
    255). Applies the LeWM resize + ImageNet normalization and returns the CLS
    token of the last hidden state.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = LEWM_LATENT_DIM,
        image_size: tuple[int, int] = tuple(LEWM_IMAGE_SIZE),
        image_mean: list[float] = LEWM_IMAGE_MEAN,
        image_std: list[float] = LEWM_IMAGE_STD,
        use_imagenet_normalization: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.resize = T.Resize(tuple(image_size), antialias=True)
        self.normalize = T.Normalize(mean=image_mean, std=image_std)
        self.use_imagenet_normalization = use_imagenet_normalization
        # Keep the wrapper frozen; the raw encoder is already frozen in the loader.
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        device: torch.device | str = "cpu",
        checkpoint_path: str | Path | None = None,
        *,
        latent_dim: int = LEWM_LATENT_DIM,
        use_imagenet_normalization: bool = True,
    ) -> "LeWMLatentEncoder":
        encoder = load_lewm_encoder(device, checkpoint_path)
        wrapper = cls(
            encoder,
            latent_dim=latent_dim,
            use_imagenet_normalization=use_imagenet_normalization,
        )
        return wrapper.to(device)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        pixels = images.to(dtype=torch.float32)
        # uint8 tensors, or floats that look like 0..255, get scaled to 0..1.
        if images.dtype == torch.uint8 or pixels.max() > 1.5:
            pixels = pixels / 255.0
        pixels = self.resize(pixels)
        if self.use_imagenet_normalization:
            pixels = self.normalize(pixels)
        return pixels

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` uint8/float pixels -> ``[B, latent_dim]`` CLS latents."""
        pixels = self._preprocess(images)
        outputs = self.encoder(pixels)
        return outputs.last_hidden_state[:, 0, :]
