import torch
import torch.nn as nn

from src.bc.dataset import PUSHT_STATE_FEATURE_DIM
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy


class StateBCPolicy(nn.Module):
    """The latent BC head reading ground-truth state instead of a learned latent.

    Input normalization lives here as buffers rather than in the dataset, so a
    saved checkpoint is self-contained: evaluation restores the exact statistics
    training used without reconstructing them from the expert data.
    """

    def __init__(
        self,
        feature_dim=PUSHT_STATE_FEATURE_DIM,
        frame_stack=3,
        action_dim=2,
        hidden_dim=256,
        action_chunk_size=5,
        feature_mean=None,
        feature_std=None,
        eps=1e-6,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        mean = (
            torch.zeros(feature_dim)
            if feature_mean is None
            else torch.as_tensor(feature_mean, dtype=torch.float32).clone()
        )
        std = (
            torch.ones(feature_dim)
            if feature_std is None
            else torch.as_tensor(feature_std, dtype=torch.float32).clone()
        )
        if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
            raise ValueError(
                f"normalization statistics must have shape ({feature_dim},); "
                f"got mean {tuple(mean.shape)} and std {tuple(std.shape)}"
            )
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", torch.clamp(std, min=eps))

        self.head = LatentBCPolicy(
            latent_dim=feature_dim,
            frame_stack=frame_stack,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_chunk_size=action_chunk_size,
        )

    def forward(self, stacked_features):
        normalized = (stacked_features - self.feature_mean) / self.feature_std
        return self.head(normalized)
