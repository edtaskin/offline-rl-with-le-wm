"""Frozen dense reward classifier used for optional LeWM PPO reward shaping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.hf_hub import resolve_artifact


class DenseRewardClassifier(nn.Module):
    """Architecture saved by scripts/probes/train_dense_reward.py."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        monotonic_outputs: bool = False,
    ):
        super().__init__()
        self.monotonic_outputs = monotonic_outputs
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if not self.monotonic_outputs or logits.shape[-1] <= 1:
            return logits
        probs = torch.sigmoid(logits)
        probs = torch.cummax(probs, dim=-1).values
        eps = torch.finfo(probs.dtype).eps
        return torch.logit(probs.clamp(eps, 1.0 - eps))


def parse_dense_reward_weights(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if not parts:
            raise ValueError("dense_reward_weights must contain at least one value")
        return [float(part) for part in parts]
    return [float(part) for part in value]


@dataclass(frozen=True)
class DenseRewardStats:
    reward_mean: float
    score_mean: float
    prob_means: list[float]


class DenseRewardShaper(nn.Module):
    """Map projected LeWM dynamics latents to a frozen classifier potential.

    The classifier checkpoints from ``scripts/probes/train_dense_reward.py``
    are trained on ``wm.encode(...)[\"emb\"][:, 0]`` and imagined
    ``wm.predict(...)[:, -1]`` latents. Do not feed raw CLS policy latents here.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        weights: str | list[float] | tuple[float, ...],
        scale: float = 1.0,
        clip: float = 1.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        checkpoint = torch.load(
            resolve_artifact(checkpoint_path), map_location=device, weights_only=False
        )
        output_dim = int(checkpoint["output_dim"])
        parsed_weights = parse_dense_reward_weights(weights)
        if len(parsed_weights) != output_dim:
            raise ValueError(
                f"dense_reward_weights has {len(parsed_weights)} values, "
                f"but checkpoint has {output_dim} heads"
            )

        self.model = DenseRewardClassifier(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=output_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            depth=int(checkpoint["depth"]),
            monotonic_outputs=bool(checkpoint.get("monotonic_outputs", False)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.requires_grad_(False)

        x_mean = torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device)
        x_std = torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device)
        weights_t = torch.as_tensor(parsed_weights, dtype=torch.float32, device=device)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std.clamp_min(1e-6))
        self.register_buffer("weights", weights_t)
        self.scale = float(scale)
        self.clip = float(clip)
        self.horizons = [int(h) for h in checkpoint.get("horizons", range(1, output_dim + 1))]
        self.frameskip = int(checkpoint.get("frameskip", 5))

    @torch.no_grad()
    def score(self, projected_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted score and per-head probabilities for projected latents."""
        if projected_latents.ndim == 3:
            z = projected_latents[:, -1, :]
        elif projected_latents.ndim == 2:
            z = projected_latents
        else:
            raise ValueError(
                f"expected [B,F,D] or [B,D] latents, got {tuple(projected_latents.shape)}"
            )
        if z.shape[-1] != self.x_mean.shape[-1]:
            raise ValueError(
                f"dense reward classifier expects latent dim {self.x_mean.shape[-1]}, "
                f"got {z.shape[-1]}"
            )
        x = (z - self.x_mean) / self.x_std
        probs = torch.sigmoid(self.model(x))
        score = (probs * self.weights).sum(dim=-1)
        return score, probs

    @torch.no_grad()
    def reward(
        self,
        current_score: torch.Tensor,
        previous_score: torch.Tensor,
        *,
        mode: str = "potential",
        discount: float = 1.0,
        positive_only: bool = False,
    ) -> torch.Tensor:
        if mode == "potential":
            raw = float(discount) * current_score - previous_score
        elif mode == "delta":
            raw = current_score - previous_score
        elif mode == "score":
            raw = current_score
        else:
            raise ValueError(f"dense reward mode must be 'potential', 'delta' or 'score', got {mode!r}")
        if positive_only:
            raw = raw.clamp_min(0.0)
        shaped = raw * self.scale
        if self.clip > 0:
            shaped = shaped.clamp(-self.clip, self.clip)
        return shaped
