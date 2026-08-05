"""Frozen map from LeWM's projected dynamics latent back to the raw CLS latent.

LeWM's predictor emits latents in the *projected* space (``projector(cls)``),
while the BC/PPO policies consume the raw ViT CLS token. Imagination has to
bridge that gap somehow: ``src/ppo/train_lewm.py`` has done it through pixels
(decode the imagined latent to a frame, re-encode the frame with the ViT). A
de-projector does the same job directly in latent space -- no image
reconstruction, no ViT pass over synthetic frames, ~three orders of magnitude
less compute per imagined step.

It is trained offline from the expert dataset (see
``scripts/deprojector/train_deprojector_pusht.py``) and consumes zero
environment interaction, exactly like the decoder and the probes.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DEPROJECTOR_CHECKPOINT_HF = (
    "hf://offline-rl-with-le-wm/deprojector_lewm_pusht/deprojector.pt"
)
DEPROJECTOR_LATENT_DIM = 192
DEPROJECTOR_HIDDEN_DIM = 2048
DEPROJECTOR_DEPTH = 2


class Deprojector(nn.Module):
    """``projected [B, D] -> raw CLS [B, D]``.

    Mirrors the shape of LeWM's own projector (wide MLP with BatchNorm), which
    is the map this is approximately inverting.
    """

    def __init__(
        self,
        latent_dim: int = DEPROJECTOR_LATENT_DIM,
        hidden_dim: int = DEPROJECTOR_HIDDEN_DIM,
        depth: int = DEPROJECTOR_DEPTH,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    @property
    def config(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
        }

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 2:
            raise ValueError(f"expected [B, D] projected latents, got {tuple(projected.shape)}")
        return self.net(projected)

    def train(self, mode: bool = True):
        """Stay in inference mode once frozen.

        The BatchNorm statistics are calibrated on the imagined-latent
        distribution at training time; a nested ``train()`` from a policy or
        trainer must not start updating them against dream batches.
        """
        if getattr(self, "_frozen", False):
            return super().train(False)
        return super().train(mode)

    def freeze(self):
        self._frozen = True
        self.eval()
        self.requires_grad_(False)
        return self


def save_deprojector(model: Deprojector, path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": model.config,
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def resolve_deprojector_reference(reference) -> Path:
    """Accept a Hub reference, an absolute path, or a repo-relative path."""
    from src.utils.hf_hub import parse_hf_artifact_reference, resolve_artifact

    text = str(reference)
    if parse_hf_artifact_reference(text) is not None:
        return resolve_artifact(text)
    path = Path(reference)
    if not path.is_absolute() and not path.is_file():
        candidate = Path(__file__).resolve().parents[2] / path
        if candidate.is_file():
            return candidate
    return path


def load_deprojector(path, device: torch.device | str = "cpu") -> Deprojector:
    """Load a frozen de-projector saved by :func:`save_deprojector`.

    ``path`` may be a local path or an ``hf://owner/repo/file.pt`` reference.
    """
    path = resolve_deprojector_reference(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing de-projector checkpoint: {path}. Train one with "
            "scripts/deprojector/train_deprojector_pusht.py, or point at the "
            f"published one ({DEPROJECTOR_CHECKPOINT_HF})."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Deprojector(**payload["config"])
    model.load_state_dict(payload["model"])
    return model.to(device).freeze()
