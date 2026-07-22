"""Frozen encoder adapters scoped to the visual-robustness ablation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.representations.lewm import LeWMEncoder, default_lewm_checkpoint_path


IMAGE_SIZE = (224, 224)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class DINOv2Encoder(nn.Module):
    feature_dim = 384
    latent_dim = 384

    def __init__(self, model, device, checkpoint_path, repo_path):
        super().__init__()
        from torchvision.transforms import Normalize, Resize

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.checkpoint_path = Path(checkpoint_path)
        self.repo_path = Path(repo_path)
        self.resize = Resize(IMAGE_SIZE, antialias=True)
        self.normalize = Normalize(IMAGE_MEAN, IMAGE_STD)

    @classmethod
    def load(cls, device="cpu", checkpoint_path=None, repo_path=None):
        hub_dir = Path(torch.hub.get_dir())
        checkpoint_path = Path(
            checkpoint_path or hub_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
        )
        repo_path = Path(repo_path or hub_dir / "facebookresearch_dinov2_main")
        if not repo_path.joinpath("hubconf.py").exists():
            raise FileNotFoundError(f"local DINOv2 repository not found at {repo_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"local DINOv2 checkpoint not found at {checkpoint_path}")
        model = torch.hub.load(
            str(repo_path), "dinov2_vits14", source="local", pretrained=False
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return cls(model, device, checkpoint_path, repo_path)

    @property
    def preprocessing_metadata(self):
        return {
            "image_size": list(IMAGE_SIZE),
            "image_normalization": "imagenet",
            "image_mean": list(IMAGE_MEAN),
            "image_std": list(IMAGE_STD),
        }

    def _preprocess(self, images):
        pixels = images.to(device=self.device, dtype=torch.float32)
        if images.dtype == torch.uint8 or pixels.max() > 1.5:
            pixels = pixels / 255.0
        return self.normalize(self.resize(pixels))

    @torch.no_grad()
    def forward(self, images):
        features = self.model.forward_features(self._preprocess(images))
        return features["x_norm_clstoken"]

    def encode(self, images):
        return self(images)


def load_encoder(
    name,
    device="cpu",
    *,
    lewm_checkpoint=None,
    dinov2_checkpoint=None,
    dinov2_repo=None,
):
    if name == "lewm":
        return LeWMEncoder.load(
            device=device,
            checkpoint_path=lewm_checkpoint or default_lewm_checkpoint_path(),
            feature_dim=192,
        )
    if name == "dinov2":
        return DINOv2Encoder.load(
            device=device,
            checkpoint_path=dinov2_checkpoint,
            repo_path=dinov2_repo,
        )
    raise ValueError("encoder must be 'lewm' or 'dinov2'")


def encoder_metadata(name, encoder):
    return {
        "name": name,
        "feature_dim": int(encoder.feature_dim),
        "checkpoint_path": str(Path(encoder.checkpoint_path).resolve()),
        "preprocessing": dict(encoder.preprocessing_metadata),
    }

