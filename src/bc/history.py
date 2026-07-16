from collections import deque

import torch


def history_indices(index, episode_start, frame_stack, frame_stride):
    return [
        max(episode_start, index - offset * frame_stride)
        for offset in range(frame_stack - 1, -1, -1)
    ]


def action_chunk_indices(index, episode_end, action_chunk_size):
    return [min(episode_end - 1, index + offset) for offset in range(action_chunk_size)]


class FeatureHistory:
    def __init__(self, frame_stack, frame_stride):
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        max_history = (frame_stack - 1) * frame_stride + 1
        self._features = deque(maxlen=max_history)

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
            raise RuntimeError("cannot stack an empty feature history")
        features = list(self._features)
        oldest = features[0]
        selected = []
        for offset in range(self.frame_stack - 1, -1, -1):
            history_index = len(features) - 1 - offset * self.frame_stride
            selected.append(features[history_index] if history_index >= 0 else oldest)
        stacked = torch.stack(selected, dim=0).unsqueeze(0)
        return stacked.to(device) if device is not None else stacked


def temporal_ensemble_action(action_predictions, decay):
    """Average overlapping predictions for one target timestep."""
    if not action_predictions:
        raise ValueError("temporal ensemble needs at least one action prediction")
    stacked_actions = torch.stack(list(action_predictions), dim=0)
    if decay == 0.0 or len(action_predictions) == 1:
        return stacked_actions.mean(dim=0)
    prediction_age = torch.arange(
        len(action_predictions),
        dtype=stacked_actions.dtype,
        device=stacked_actions.device,
    )
    weights = torch.exp(-decay * prediction_age)
    weights = weights / weights.sum()
    return (stacked_actions * weights[:, None]).sum(dim=0)
