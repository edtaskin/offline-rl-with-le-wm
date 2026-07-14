"""Latent PPO actor-critic that fine-tunes a Latent BC policy.

Architecture (per observation)::

    image history [B, frame_stack, C, H, W]
        -> frozen LeWM encoder (shared)         -> stacked latents [B, frame_stack, 192]
        -> LatentBCPolicy                        -> action mean     [B, k, action_dim]
        -> diagonal Gaussian (state-independent log-std)
        -> sampled action chunk                  [B, k, action_dim]

The BC policy provides the action mean and is initialized from a trained BC
checkpoint (``build_latent_agent``); PPO fine-tunes it together with the
exploration ``log_std`` and a separate value head. The encoder is frozen, so a
stacked latent is a fixed function of the images: the trainer/evaluator encode
each frame once and every method here operates on precomputed
``[B, frame_stack, 192]`` latents (``*_from_latents``) -- the ViT never appears
in the optimization loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy


class LatentPPOActor(nn.Module):
    def __init__(
        self,
        encoder,
        bc_policy,
        action_dim: int = 2,
        action_chunk_size: int = 1,
        init_log_std: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.bc_policy = bc_policy
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        # Freeze encoder, like in BC training.
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # PPO exploration noise: state-independent, learned log-std.
        self.log_std = nn.Parameter(
            torch.full((action_chunk_size, action_dim), float(init_log_std))
        )

    def dist_from_latents(self, stacked_latents: torch.Tensor) -> Normal:
        # BC policy gives the deterministic action-chunk mean.
        action_mean = self.bc_policy(stacked_latents)
        # PPO turns that into a stochastic policy.
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        return Normal(action_mean, action_std)

class LatentCritic(nn.Module):
    def __init__(
        self,
        encoder,
        latent_dim: int = 192,
        frame_stack: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        input_dim = latent_dim * frame_stack
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def value_from_latents(self, stacked_latents: torch.Tensor) -> torch.Tensor:
        flat_latents = stacked_latents.reshape(stacked_latents.shape[0], -1)
        return self.value_net(flat_latents).squeeze(-1)


class LatentPPOAgent(nn.Module):
    def __init__(
        self,
        encoder,
        bc_policy,
        latent_dim: int = 192,
        frame_stack: int = 3,
        action_dim: int = 2,
        action_chunk_size: int = 1,
        hidden_dim: int = 256,
        init_log_std: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.frame_stack = frame_stack
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        self.actor = LatentPPOActor(
            encoder=encoder,
            bc_policy=bc_policy,
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            init_log_std=init_log_std,
        )
        self.critic = LatentCritic(
            encoder=encoder,
            latent_dim=latent_dim,
            frame_stack=frame_stack,
            hidden_dim=hidden_dim,
        )

    # ---- latent fast-path (used by the PPO update / rollout) ----
    def get_action_and_value_from_latents(
        self, stacked_latents: torch.Tensor, action: torch.Tensor | None = None
    ):
        dist = self.actor.dist_from_latents(stacked_latents)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=(-1, -2))
        entropy = dist.entropy().sum(dim=(-1, -2))
        value = self.critic.value_from_latents(stacked_latents)
        return action, logprob, entropy, value

    def get_value_from_latents(self, stacked_latents: torch.Tensor) -> torch.Tensor:
        return self.critic.value_from_latents(stacked_latents)


def build_latent_agent(
    *,
    encoder,
    latent_dim: int = 192,
    frame_stack: int = 3,
    action_dim: int = 2,
    action_chunk_size: int = 5,
    hidden_dim: int = 256,
    init_log_std: float = 0.0,
    bc_checkpoint_path: str | None = None,
    device: torch.device | str = "cpu",
) -> LatentPPOAgent:
    """Construct a :class:`LatentPPOAgent`, optionally loading BC policy weights.

    ``encoder`` should return ``[B, latent_dim]`` latents for ``[B, C, H, W]``
    images (e.g. :class:`src.ppo.lewm_encoder.LeWMLatentEncoder`). When
    ``bc_checkpoint_path`` is given, its weights are loaded into the actor's
    ``bc_policy`` so PPO starts from the BC prior.
    """
    bc_policy = LatentBCPolicy(
        latent_dim=latent_dim,
        frame_stack=frame_stack,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_chunk_size=action_chunk_size,
    )
    if bc_checkpoint_path is not None:
        state_dict = torch.load(bc_checkpoint_path, map_location=device)
        bc_policy.load_state_dict(state_dict)

    agent = LatentPPOAgent(
        encoder=encoder,
        bc_policy=bc_policy,
        latent_dim=latent_dim,
        frame_stack=frame_stack,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        init_log_std=init_log_std,
    )
    return agent.to(device)
