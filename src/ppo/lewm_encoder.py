"""Compatibility exports for the shared LeWM representation module."""

from src.representations.lewm import (
    LEWM_DEFAULT_FEATURE_DIM as LEWM_LATENT_DIM,
    LeWMEncoder,
    default_lewm_checkpoint_path,
    load_lewm_encoder,
)

LeWMLatentEncoder = LeWMEncoder

__all__ = [
    "LEWM_LATENT_DIM",
    "LeWMEncoder",
    "LeWMLatentEncoder",
    "default_lewm_checkpoint_path",
    "load_lewm_encoder",
]
