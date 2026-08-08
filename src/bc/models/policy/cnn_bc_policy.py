"""From-scratch pixel encoder for the BC encoder baseline.

The comparison this serves is *frozen pretrained features vs end-to-end
training*, not "MLP vs CNN": this trunk receives gradients from the action loss
and the LeWM ViT does not. The output width is pinned to LeWM's 192-d CLS so the
head, its width, and the history contract are identical across arms and only the
source of the 192 numbers differs.

Spatial softmax follows the standard visuomotor recipe (Levine et al., and the
PushT baselines in Diffusion Policy): PushT is a positional task, and average
pooling discards exactly the spatial information the policy needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy


# ImageNet statistics, matching what the frozen LeWM encoder normalizes with, so
# the two arms see the same input distribution.
CNN_IMAGE_MEAN = (0.485, 0.456, 0.406)
CNN_IMAGE_STD = (0.229, 0.224, 0.225)
CNN_DEFAULT_FEATURE_DIM = 192


class SpatialSoftmax(nn.Module):
    """Expected 2-d keypoint coordinates per channel.

    Returns ``2 * channels`` values in [-1, 1]: for each feature map, the
    softmax-weighted mean of the normalized pixel grid.
    """

    def __init__(self, channels, height, width, temperature=1.0):
        super().__init__()
        self.channels = channels
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))

    def forward(self, features):
        batch, channels = features.shape[:2]
        flat = features.reshape(batch, channels, -1) / self.temperature
        attention = F.softmax(flat, dim=-1)
        keypoint_x = (attention * self.pos_x).sum(dim=-1)
        keypoint_y = (attention * self.pos_y).sum(dim=-1)
        return torch.cat([keypoint_x, keypoint_y], dim=-1)


class CNNEncoder(nn.Module):
    """Conv trunk mapping one RGB frame to a ``feature_dim`` embedding.

    GroupNorm rather than BatchNorm: batches mix frames from the same episode,
    and evaluation runs a single environment, so batch statistics would differ
    between training and rollout.
    """

    def __init__(
        self,
        feature_dim=CNN_DEFAULT_FEATURE_DIM,
        input_resolution=224,
        channels=(32, 64, 128, 256),
        num_keypoints=32,
        groups=8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_resolution = input_resolution

        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(groups, out_channels), out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        # A 1x1 projection sets the keypoint count independently of trunk width.
        layers.append(nn.Conv2d(in_channels, num_keypoints, kernel_size=1))
        self.trunk = nn.Sequential(*layers)

        spatial = input_resolution
        for _ in channels:
            spatial = (spatial + 1) // 2
        self.spatial_softmax = SpatialSoftmax(num_keypoints, spatial, spatial)
        self.project = nn.Linear(2 * num_keypoints, feature_dim)

        self.register_buffer("image_mean", torch.tensor(CNN_IMAGE_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(CNN_IMAGE_STD).reshape(1, 3, 1, 1))

    def normalize(self, images):
        # Driven by dtype, never by pixel values. A value-dependent branch such
        # as ``images.max() > 1.5`` silently rescales differently for a dark
        # frame than a bright one, which would make training and evaluation
        # disagree on an input the policy has to treat identically.
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
        return (images - self.image_mean) / self.image_std

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) images, got shape {tuple(images.shape)}")
        features = self.trunk(self.normalize(images))
        return self.project(self.spatial_softmax(features))


class CNNBCPolicy(nn.Module):
    """End-to-end pixel BC: per-frame conv encoder plus the shared chunked head."""

    def __init__(
        self,
        feature_dim=CNN_DEFAULT_FEATURE_DIM,
        frame_stack=3,
        action_dim=2,
        hidden_dim=256,
        action_chunk_size=5,
        input_resolution=224,
        num_keypoints=32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.frame_stack = frame_stack
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.encoder = CNNEncoder(
            feature_dim=feature_dim,
            input_resolution=input_resolution,
            num_keypoints=num_keypoints,
        )
        self.head = LatentBCPolicy(
            latent_dim=feature_dim,
            frame_stack=frame_stack,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_chunk_size=action_chunk_size,
        )

    def encode_stack(self, stacked_images):
        """Encode ``(B, F, C, H, W)`` frames into ``(B, F, feature_dim)``."""

        if stacked_images.ndim != 5:
            raise ValueError(
                f"expected (B, F, C, H, W) frames, got shape {tuple(stacked_images.shape)}"
            )
        batch, frames = stacked_images.shape[:2]
        flat = stacked_images.reshape(batch * frames, *stacked_images.shape[2:])
        return self.encoder(flat).reshape(batch, frames, self.feature_dim)

    def forward(self, stacked_images):
        return self.head(self.encode_stack(stacked_images))
