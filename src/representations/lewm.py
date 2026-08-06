"""Shared frozen LeWM image encoder used by BC, PPO, and evaluation."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


LEWM_IMAGE_SIZE = (224, 224)
LEWM_IMAGE_MEAN = [0.485, 0.456, 0.406]
LEWM_IMAGE_STD = [0.229, 0.224, 0.225]
LEWM_IMAGE_NORMALIZATION = "imagenet"
LEWM_DEFAULT_FEATURE_DIM = 192


def _ensure_lewm_source_path() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    env_path = os.getenv("LE_WM_PATH")
    repo_candidate = Path(__file__).resolve().parents[2] / "le-wm"
    candidate = Path(env_path) if env_path else repo_candidate
    if candidate.exists() and str(candidate.resolve()) not in sys.path:
        sys.path.insert(0, str(candidate.resolve()))


def load_stable_worldmodel():
    _ensure_lewm_source_path()
    return importlib.import_module("stable_worldmodel")


def default_lewm_checkpoint_path() -> Path:
    swm = load_stable_worldmodel()
    return Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")


def lewm_preprocessing_metadata(normalization=LEWM_IMAGE_NORMALIZATION):
    return {
        "image_size": LEWM_IMAGE_SIZE,
        "image_normalization": normalization,
        "image_mean": LEWM_IMAGE_MEAN,
        "image_std": LEWM_IMAGE_STD,
    }


def load_lewm_world_model(
    checkpoint="hf_pusht/weights.pt",
    cache_dir=None,
    device="cpu",
    logger=None,
) -> nn.Module:
    """Load the full (frozen) LeWM world model: encoder, projector, predictor.

    Tries ``swm.wm.utils.load_pretrained`` first, the loading path the decoder
    and probe scripts were written against. On transformers >= 5 the published
    ``weights.pt`` fails its strict load because the ViT submodule names changed
    (``encoder.encoder.layer.N.attention.attention.query`` became
    ``encoder.layers.N.attention.q_proj``); in that case fall back to the
    converted object checkpoint that ``scripts/download_lewm_checkpoint.py``
    saves with the keys remapped. The weights are identical either way.
    """
    swm = load_stable_worldmodel()
    cache_dir = str(cache_dir or Path(__file__).resolve().parents[2] / "le-wm/models")
    try:
        model = swm.wm.utils.load_pretrained(checkpoint, cache_dir=cache_dir)
    except (RuntimeError, FileNotFoundError) as exc:
        object_path = default_lewm_checkpoint_path()
        if not object_path.exists():
            raise FileNotFoundError(
                f"Could not load LeWM weights via load_pretrained ({exc}) and no "
                f"object checkpoint at {object_path}. Run "
                "`python -m scripts.download_lewm_checkpoint` first."
            ) from exc
        message = (
            f"load_pretrained({checkpoint!r}) failed ({type(exc).__name__}); "
            f"falling back to the converted object checkpoint {object_path}"
        )
        if logger is not None:
            logger.warning("%s", message)
        else:
            print(f"warning: {message}")
        model = torch.load(object_path, map_location="cpu", weights_only=False)

    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def load_lewm_encoder(device="cpu", checkpoint_path=None) -> nn.Module:
    load_stable_worldmodel()
    checkpoint_path = Path(checkpoint_path or default_lewm_checkpoint_path())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"LeWM encoder checkpoint not found at {checkpoint_path}. "
            "Run `python -m scripts.download_lewm_checkpoint` or pass an explicit path."
        )
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = model.encoder.to(device).eval()
    encoder.requires_grad_(False)
    return encoder


class LeWMEncoder(nn.Module):
    """Frozen LeWM preprocessing and CLS extraction behind one module."""

    def __init__(
        self,
        encoder,
        device="cpu",
        checkpoint_path=None,
        feature_dim=None,
        latent_dim=None,
        normalization=LEWM_IMAGE_NORMALIZATION,
        image_size=LEWM_IMAGE_SIZE,
        image_mean=LEWM_IMAGE_MEAN,
        image_std=LEWM_IMAGE_STD,
    ):
        super().__init__()
        try:
            import torchvision.transforms as transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for LeWM image preprocessing. "
                "Install torchvision or use the intended project environment."
            ) from exc
        if normalization != LEWM_IMAGE_NORMALIZATION:
            raise ValueError(
                f"unsupported LeWM image normalization: {normalization!r}. The frozen "
                "encoder was trained under ImageNet statistics and only "
                f"{LEWM_IMAGE_NORMALIZATION!r} reproduces its training distribution."
            )
        resolved_dim = latent_dim if latent_dim is not None else feature_dim
        self.latent_dim = int(resolved_dim or LEWM_DEFAULT_FEATURE_DIM)
        self.feature_dim = self.latent_dim
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.normalization = normalization
        self.image_size = tuple(image_size)
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.encoder = encoder.to(self.device).eval()
        self.encoder.requires_grad_(False)
        self.resize = transforms.Resize(self.image_size, antialias=True)
        self.normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)

    @classmethod
    def from_checkpoint(
        cls,
        device="cpu",
        checkpoint_path=None,
        *,
        latent_dim=LEWM_DEFAULT_FEATURE_DIM,
        normalization=LEWM_IMAGE_NORMALIZATION,
    ):
        checkpoint_path = Path(checkpoint_path or default_lewm_checkpoint_path())
        print("Loading official LeWM object checkpoint...")
        print(checkpoint_path)
        encoder = load_lewm_encoder(device, checkpoint_path)
        wrapper = cls(
            encoder=encoder,
            device=device,
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            normalization=normalization,
        )
        print("Successfully loaded and frozen the LeWM Encoder from the official checkpoint!")
        return wrapper

    @classmethod
    def load(
        cls,
        device="cpu",
        checkpoint_path=None,
        feature_dim=LEWM_DEFAULT_FEATURE_DIM,
        normalization=LEWM_IMAGE_NORMALIZATION,
    ):
        return cls.from_checkpoint(
            device=device,
            checkpoint_path=checkpoint_path,
            latent_dim=feature_dim,
            normalization=normalization,
        )

    @property
    def preprocessing_metadata(self):
        return {
            "image_size": self.image_size,
            "image_normalization": self.normalization,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Scale, resize and ImageNet-normalize, matching how LeWM was trained.

        Upstream normalizes before resizing (``le-wm/eval.py`` composes
        ``ToDtype(scale=True) -> Normalize(ImageNet) -> Resize``); the two orders
        are numerically identical because antialiased resize weights sum to one.
        Unlike ``ToDtype(scale=True)``, which only rescales integer dtypes, float
        batches still in 0-255 are scaled here as well.
        """
        if images.ndim != 4:
            raise ValueError(f"expected NCHW image batch, got shape {tuple(images.shape)}")
        pixels = images.to(device=self.device, dtype=torch.float32)
        if images.dtype == torch.uint8 or pixels.max() > 1.5:
            pixels = pixels / 255.0
        return self.normalize(self.resize(pixels))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # interpolate_pos_encoding mirrors JEPA.encode; a no-op at the native 224
        # but required for any other image_size.
        outputs = self.encoder(self._preprocess(images), interpolate_pos_encoding=True)
        features = outputs.last_hidden_state[:, 0, :]
        if features.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected LeWM latent_dim={self.latent_dim}, got {features.shape[-1]}"
            )
        return features

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self(images)
