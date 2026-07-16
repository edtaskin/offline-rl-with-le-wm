import importlib
import os
import sys
from pathlib import Path

import torch


LEWM_IMAGE_SIZE = (224, 224)
LEWM_IMAGE_MEAN = [0.485, 0.456, 0.406]
LEWM_IMAGE_STD = [0.229, 0.224, 0.225]
LEWM_IMAGE_NORMALIZATION = "imagenet"
LEWM_LEGACY_IMAGE_NORMALIZATION = "legacy_div255"
LEWM_DEFAULT_FEATURE_DIM = 192


def load_stable_worldmodel():
    lewm_path = os.getenv("LE_WM_PATH")
    if lewm_path is None:
        raise ValueError("LE_WM_PATH environment variable not set")
    if lewm_path not in sys.path:
        sys.path.insert(0, lewm_path)
    return importlib.import_module("stable_worldmodel")


def default_lewm_checkpoint_path():
    swm = load_stable_worldmodel()
    return Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")


def lewm_preprocessing_metadata(normalization=LEWM_IMAGE_NORMALIZATION):
    return {
        "image_size": LEWM_IMAGE_SIZE,
        "image_normalization": normalization,
        "image_mean": LEWM_IMAGE_MEAN,
        "image_std": LEWM_IMAGE_STD,
    }


class LeWMFeatureExtractor:
    def __init__(
        self,
        encoder,
        device,
        checkpoint_path,
        feature_dim=LEWM_DEFAULT_FEATURE_DIM,
        normalization=LEWM_IMAGE_NORMALIZATION,
    ):
        try:
            import torchvision.transforms as transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for LeWM image preprocessing. "
                "Install torchvision or use the intended project environment."
            ) from exc

        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.feature_dim = int(feature_dim)
        self.normalization = normalization
        self.encoder = encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self._resize = transforms.Resize(LEWM_IMAGE_SIZE, antialias=True)
        self._normalize = transforms.Normalize(mean=LEWM_IMAGE_MEAN, std=LEWM_IMAGE_STD)

    @classmethod
    def load(
        cls,
        device,
        checkpoint_path=None,
        feature_dim=LEWM_DEFAULT_FEATURE_DIM,
        normalization=LEWM_IMAGE_NORMALIZATION,
    ):
        # Register the custom JEPA classes used by the serialized checkpoint.
        load_stable_worldmodel()
        checkpoint_path = Path(checkpoint_path or default_lewm_checkpoint_path())
        print("Loading official LeWM object checkpoint...")
        print(checkpoint_path)
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        extractor = cls(
            encoder=model.encoder,
            device=device,
            checkpoint_path=checkpoint_path,
            feature_dim=feature_dim,
            normalization=normalization,
        )
        print("Successfully loaded and frozen the LeWM Encoder from the official checkpoint!")
        return extractor

    @property
    def preprocessing_metadata(self):
        return lewm_preprocessing_metadata(self.normalization)

    def encode(self, images):
        if images.ndim != 4:
            raise ValueError(f"expected NCHW image batch, got shape {tuple(images.shape)}")
        images = images.to(self.device, dtype=torch.float32)
        images = self._resize(images)
        if self.normalization == LEWM_IMAGE_NORMALIZATION:
            images = self._normalize(images)
        elif self.normalization != LEWM_LEGACY_IMAGE_NORMALIZATION:
            raise ValueError(f"unsupported LeWM image normalization: {self.normalization}")
        # no_grad keeps outputs usable as frozen inputs to a trainable BC head.
        with torch.no_grad():
            outputs = self.encoder(images)
            features = outputs.last_hidden_state[:, 0, :]
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected LeWM feature_dim={self.feature_dim}, got {features.shape[-1]}"
            )
        return features
