"""Actor-critic network for continuous-control PPO.

A standard CleanRL-style MLP: separate policy and value trunks, a diagonal
Gaussian policy with a state-independent (learned) log-std, and orthogonal
initialization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers.append(layer_init(nn.Linear(last, h)))
        layers.append(nn.Tanh())
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.critic = _mlp(obs_dim, hidden_dims, 1, out_std=1.0)
        self.actor_mean = _mlp(obs_dim, hidden_dims, act_dim, out_std=0.01)
        # State-independent log-std, initialized to std = 1.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return action, logprob, entropy, value
