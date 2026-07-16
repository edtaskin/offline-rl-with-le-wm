"""Small utilities: seeding, device selection, and running mean/std trackers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def get_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is available on Apple Silicon but tends to be slower than CPU for the
    # tiny MLPs used here, so we prefer CPU unless CUDA is present.
    return torch.device("cpu")


class RunningMeanStd:
    """Welford-style running mean/variance over a batch axis (Numpy).

    Used for return normalization, mirroring the behavior of
    ``gymnasium.wrappers.NormalizeReward`` but shared across all parallel
    environments.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count


class RewardNormalizer:
    """Scales rewards by the std of the discounted return (gym-style).

    Tracks a running estimate of the discounted return per environment and
    divides rewards by its standard deviation. Returns are reset to zero when an
    episode ends.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(shape=())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.float64)
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        out = rewards / np.sqrt(self.rms.var + self.epsilon)
        self.returns = self.returns * (1.0 - dones)  # reset finished episodes
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"var": self.rms.var, "mean": self.rms.mean, "count": self.rms.count}

