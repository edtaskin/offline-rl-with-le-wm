"""Compatibility exports for the shared LeWM representation module."""

from src.representations.lewm import (
    LEWM_DEFAULT_FEATURE_DIM,
    LEWM_IMAGE_MEAN,
    LEWM_IMAGE_NORMALIZATION,
    LEWM_IMAGE_SIZE,
    LEWM_IMAGE_STD,
    LeWMEncoder,
    default_lewm_checkpoint_path,
    lewm_preprocessing_metadata,
    load_lewm_encoder,
    load_stable_worldmodel,
)

LeWMFeatureExtractor = LeWMEncoder

__all__ = [
    "LEWM_DEFAULT_FEATURE_DIM",
    "LEWM_IMAGE_MEAN",
    "LEWM_IMAGE_NORMALIZATION",
    "LEWM_IMAGE_SIZE",
    "LEWM_IMAGE_STD",
    "LeWMEncoder",
    "LeWMFeatureExtractor",
    "default_lewm_checkpoint_path",
    "lewm_preprocessing_metadata",
    "load_lewm_encoder",
    "load_stable_worldmodel",
]
