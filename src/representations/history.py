"""Shared dilated latent history for training and evaluation."""

from collections import deque

import torch


class LatentHistory:
    def __init__(self, frame_stack: int, frame_stride: int):
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        max_length = (frame_stack - 1) * frame_stride + 1
        self._features = deque(maxlen=max_length)

    def __len__(self):
        return len(self._features)

    def clear(self):
        self._features.clear()

    def append(self, feature):
        if feature.ndim == 2:
            if feature.shape[0] != 1:
                raise ValueError("batched history features must have batch size 1")
            feature = feature.squeeze(0)
        if feature.ndim != 1:
            raise ValueError(f"expected a feature vector, got shape {tuple(feature.shape)}")
        self._features.append(feature)

    def stacked(self, device=None):
        if not self._features:
            raise RuntimeError("cannot stack an empty latent history")
        features = list(self._features)
        oldest = features[0]
        selected = []
        for offset in range(self.frame_stack - 1, -1, -1):
            index = len(features) - 1 - offset * self.frame_stride
            selected.append(features[index] if index >= 0 else oldest)
        stacked = torch.stack(selected, dim=0)
        return stacked.to(device) if device is not None else stacked


def temporal_ensemble_action(predictions, decay):
    if not predictions:
        raise ValueError("temporal ensemble needs at least one action prediction")
    stacked = torch.stack(list(predictions), dim=0)
    if decay == 0.0 or len(predictions) == 1:
        return stacked.mean(dim=0)
    age = torch.arange(len(predictions), dtype=stacked.dtype, device=stacked.device)
    weights = torch.exp(-decay * age)
    weights = weights / weights.sum()
    return (stacked * weights[:, None]).sum(dim=0)
