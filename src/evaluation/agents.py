"""Method-specific checkpoint loaders behind a common per-step agent interface."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import numpy as np
import torch

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.ppo.agent import build_latent_agent
from src.representations.history import LatentHistory, temporal_ensemble_action
from src.representations.lewm import (
    LEWM_IMAGE_NORMALIZATION,
    LEWM_LEGACY_IMAGE_NORMALIZATION,
    LeWMEncoder,
)
from src.utils.hf_hub import (
    HubArtifactReference,
    parse_hf_artifact_reference,
    resolve_artifact,
    resolve_artifacts,
)


_PPO_ENCODER_PREFIXES = ("actor.encoder.", "critic.encoder.")


def resolve_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentChunkAgent:
    def __init__(
        self,
        *,
        agent_type,
        encoder,
        predict_chunk: Callable[[torch.Tensor, bool], torch.Tensor],
        contract,
        device,
        deterministic=True,
        execution_mode="open-loop",
        replan_interval=1,
        temporal_ensemble_decay=0.01,
        metadata=None,
    ):
        if execution_mode not in ("open-loop", "receding-horizon", "temporal-ensemble"):
            raise ValueError(f"unsupported execution_mode: {execution_mode}")
        if replan_interval < 1:
            raise ValueError("replan_interval must be at least 1")
        if temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        self.agent_type = agent_type
        self.encoder = encoder
        self.predict_chunk = predict_chunk
        self.contract = {key: int(value) for key, value in contract.items()}
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.execution_mode = execution_mode
        self.replan_interval = int(replan_interval)
        self.temporal_ensemble_decay = float(temporal_ensemble_decay)
        self.metadata = {
            "contract": dict(self.contract),
            "deterministic": self.deterministic,
            "execution_mode": execution_mode,
            "replan_interval": self.replan_interval,
            "temporal_ensemble_decay": self.temporal_ensemble_decay,
            **(metadata or {}),
        }
        self.history = LatentHistory(
            self.contract["frame_stack"], self.contract["frame_stride"]
        )
        self._action_queue = deque()
        self._temporal_predictions = defaultdict(list)
        self._step = 0

    def reset(self, seed):
        torch.manual_seed(seed)
        self.history.clear()
        self._action_queue.clear()
        self._temporal_predictions.clear()
        self._step = 0

    @torch.no_grad()
    def _encode_observation(self, observation):
        image = torch.as_tensor(np.asarray(observation), device=self.device)
        if image.ndim != 3:
            raise ValueError(f"expected HWC observation, got shape {tuple(image.shape)}")
        image = image.permute(2, 0, 1).contiguous().unsqueeze(0)
        return self.encoder(image)[0]

    @torch.no_grad()
    def _new_chunk(self, deterministic=None):
        stacked = self.history.stacked(self.device).unsqueeze(0)
        use_deterministic = self.deterministic if deterministic is None else deterministic
        chunk = self.predict_chunk(stacked, use_deterministic)
        chunk = torch.as_tensor(chunk, dtype=torch.float32).detach().cpu()
        expected_shape = (
            self.contract["action_chunk_size"],
            self.contract["action_dim"],
        )
        if tuple(chunk.shape) != expected_shape:
            raise ValueError(
                f"chunk predictor returned shape {tuple(chunk.shape)}; expected {expected_shape}"
            )
        return torch.clamp(chunk, -1.0, 1.0)

    def act(self, observation, info):
        self.history.append(self._encode_observation(observation))
        if self.execution_mode == "temporal-ensemble":
            chunk = self._new_chunk(deterministic=True)
            for offset, action in enumerate(chunk):
                self._temporal_predictions[self._step + offset].append(action)
            action = temporal_ensemble_action(
                self._temporal_predictions.pop(self._step),
                self.temporal_ensemble_decay,
            )
            self._step += 1
            return torch.clamp(action, -1.0, 1.0).numpy()

        if not self._action_queue:
            chunk = self._new_chunk()
            execute_count = len(chunk)
            if self.execution_mode == "receding-horizon":
                execute_count = min(self.replan_interval, len(chunk))
            self._action_queue.extend(chunk[:execute_count])
        self._step += 1
        return self._action_queue.popleft().numpy()


@dataclass
class BCComponents:
    encoder: LeWMEncoder
    policy: LatentBCPolicy
    contract: dict
    stats: dict


@dataclass
class PPOComponents:
    encoder: LeWMEncoder
    agent: torch.nn.Module
    contract: dict
    config: dict


def infer_bc_stats_path(checkpoint_path):
    hub_reference = parse_hf_artifact_reference(checkpoint_path)
    if hub_reference is not None:
        path = PurePosixPath(hub_reference.filename)
        filename = f"{path.stem}_stats.pth" if path.suffix else f"{path.name}_stats.pth"
        return HubArtifactReference(
            repo_id=hub_reference.repo_id,
            filename=str(path.with_name(filename)),
        ).uri
    path = Path(checkpoint_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_stats.pth"))
    return f"{checkpoint_path}_stats.pth"


def load_bc_components(checkpoint, stats_path=None, device="auto"):
    device = resolve_device(device)
    stats_reference = stats_path or infer_bc_stats_path(checkpoint)
    checkpoint_path, resolved_stats_path = resolve_artifacts([checkpoint, stats_reference])
    stats = torch.load(resolved_stats_path, map_location="cpu")
    contract = {
        "frame_stack": int(stats.get("frame_stack", 3)),
        "frame_stride": int(stats.get("frame_stride", 1)),
        "action_chunk_size": int(stats.get("action_chunk_size", 1)),
        "latent_dim": int(stats.get("latent_dim", 192)),
        "hidden_dim": int(stats.get("hidden_dim", 256)),
        "action_dim": int(stats.get("action_dim", 2)),
    }
    normalization = stats.get("image_normalization", LEWM_LEGACY_IMAGE_NORMALIZATION)
    encoder = LeWMEncoder.from_checkpoint(
        device=device,
        latent_dim=contract["latent_dim"],
        normalization=normalization,
    )
    policy = LatentBCPolicy(
        latent_dim=contract["latent_dim"],
        frame_stack=contract["frame_stack"],
        action_dim=contract["action_dim"],
        hidden_dim=contract["hidden_dim"],
        action_chunk_size=contract["action_chunk_size"],
    ).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    return BCComponents(encoder=encoder, policy=policy, contract=contract, stats=stats)


def load_ppo_components(checkpoint, device="auto", encoder=None):
    """Load a PPO checkpoint into an evaluable agent.

    ``encoder`` lets a caller reuse an already-built frozen ViT across many
    checkpoints from the same run (the RQ1 budget curve evaluates dozens of
    snapshots); it must match ``contract['latent_dim']``. Left as ``None`` the
    encoder is rebuilt from the checkpoint's own config, as before.
    """
    device = resolve_device(device)
    checkpoint_path = resolve_artifact(checkpoint)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    contract = {
        key: int(value)
        for key, value in payload.get("contract", config).items()
        if key
        in {
            "frame_stack",
            "frame_stride",
            "action_chunk_size",
            "latent_dim",
            "hidden_dim",
            "action_dim",
        }
    }
    if encoder is None:
        encoder = LeWMEncoder.from_checkpoint(
            device=device,
            checkpoint_path=config.get("encoder_checkpoint"),
            latent_dim=contract["latent_dim"],
            normalization=config.get("image_normalization", LEWM_IMAGE_NORMALIZATION),
        )
    agent = build_latent_agent(
        encoder=encoder,
        latent_dim=contract["latent_dim"],
        frame_stack=contract["frame_stack"],
        action_dim=contract["action_dim"],
        action_chunk_size=contract["action_chunk_size"],
        hidden_dim=contract["hidden_dim"],
        init_log_std=float(config.get("init_log_std", 0.0)),
        bc_checkpoint_path=None,
        device=device,
    )
    missing, unexpected = agent.load_state_dict(payload["agent"], strict=False)
    stray = [key for key in missing if not key.startswith(_PPO_ENCODER_PREFIXES)]
    if stray or unexpected:
        raise RuntimeError(f"checkpoint mismatch (missing={stray}, unexpected={unexpected})")
    agent.eval()
    return PPOComponents(encoder=encoder, agent=agent, contract=contract, config=config)


def make_bc_evaluation_agent(
    checkpoint=None,
    stats_path=None,
    components=None,
    device="auto",
    execution_mode="open-loop",
    replan_interval=1,
    temporal_ensemble_decay=0.01,
):
    components = components or load_bc_components(checkpoint, stats_path, device)
    resolved_device = next(components.policy.parameters()).device

    def predict_chunk(stacked, deterministic):
        return components.policy(stacked)[0]

    return LatentChunkAgent(
        agent_type="bc",
        encoder=components.encoder,
        predict_chunk=predict_chunk,
        contract=components.contract,
        device=resolved_device,
        deterministic=True,
        execution_mode=execution_mode,
        replan_interval=replan_interval,
        temporal_ensemble_decay=temporal_ensemble_decay,
        metadata={
            "checkpoint": str(checkpoint) if checkpoint is not None else "in-memory",
            "stats": str(stats_path or infer_bc_stats_path(checkpoint))
            if checkpoint is not None
            else "in-memory",
        },
    )


def make_ppo_evaluation_agent(
    checkpoint=None,
    *,
    components=None,
    device="auto",
    deterministic=True,
    execution_mode="open-loop",
    replan_interval=1,
    temporal_ensemble_decay=0.01,
):
    components = components or load_ppo_components(checkpoint, device)
    resolved_device = next(components.agent.parameters()).device

    def predict_chunk(stacked, use_deterministic):
        if use_deterministic:
            return components.agent.actor.dist_from_latents(stacked).mean[0]
        action, _, _, _ = components.agent.get_action_and_value_from_latents(stacked)
        return action[0]

    return LatentChunkAgent(
        agent_type="ppo",
        encoder=components.encoder,
        predict_chunk=predict_chunk,
        contract=components.contract,
        device=resolved_device,
        deterministic=deterministic,
        execution_mode=execution_mode,
        replan_interval=replan_interval,
        temporal_ensemble_decay=temporal_ensemble_decay,
        metadata={"checkpoint": str(checkpoint) if checkpoint is not None else "in-memory"},
    )
