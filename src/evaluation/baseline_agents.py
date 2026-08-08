"""Evaluation adapters for the BC encoder baselines.

These agents are scored by the same environment-owned evaluator as every other
agent in the repository; only the observation the policy reads differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.bc.dataset import PUSHT_STATE_DIM, pusht_state_features
from src.bc.models.policy.state_bc_policy import StateBCPolicy
from src.evaluation.agents import (
    LatentChunkAgent,
    bc_training_observation_resolution,
    infer_bc_stats_path,
    resolve_device,
)
from src.utils.hf_hub import resolve_artifacts


def pusht_state_from_info(info):
    """Rebuild the recorded 5-d PushT state from the evaluation env's info dict.

    ``pos_agent`` and ``block_pose`` are both taken from the simulator's own
    ``_get_obs()``, which is exactly what the expert dataset stores, so the
    oracle reads training and evaluation states in one convention.
    """

    missing = [key for key in ("pos_agent", "block_pose") if key not in info]
    if missing:
        raise KeyError(
            f"the state oracle needs {missing} in the environment info dict; "
            "this env configuration does not expose the ground-truth state"
        )
    agent = np.asarray(info["pos_agent"], dtype=np.float32).reshape(-1)
    block = np.asarray(info["block_pose"], dtype=np.float32).reshape(-1)
    state = np.concatenate([agent[:2], block[:3]])
    if state.shape != (PUSHT_STATE_DIM,):
        raise ValueError(
            f"expected a {PUSHT_STATE_DIM}-d PushT state, got shape {state.shape}"
        )
    return state


@dataclass
class StateBCComponents:
    policy: StateBCPolicy
    contract: dict
    stats: dict


class StateChunkAgent(LatentChunkAgent):
    """Chunked BC agent whose observation is the simulator state, not pixels."""

    @torch.no_grad()
    def _encode_observation(self, observation, info=None):
        if info is None:
            raise ValueError("the state oracle requires the environment info dict")
        state = pusht_state_from_info(info)
        return pusht_state_features(torch.from_numpy(state)).to(self.device)


def load_state_bc_components(checkpoint, stats_path=None, device="auto"):
    device = resolve_device(device)
    stats_reference = stats_path or infer_bc_stats_path(checkpoint)
    checkpoint_path, resolved_stats_path = resolve_artifacts([checkpoint, stats_reference])
    stats = torch.load(resolved_stats_path, map_location="cpu")
    observation_space = stats.get("observation_space")
    if observation_space != "state":
        raise ValueError(
            f"{resolved_stats_path} records observation_space={observation_space!r}; "
            "the state oracle only loads checkpoints trained on ground-truth states"
        )
    contract = {
        "frame_stack": int(stats["frame_stack"]),
        "frame_stride": int(stats["frame_stride"]),
        "action_chunk_size": int(stats["action_chunk_size"]),
        "latent_dim": int(stats["latent_dim"]),
        "hidden_dim": int(stats["hidden_dim"]),
        "action_dim": int(stats.get("action_dim", 2)),
    }
    policy = StateBCPolicy(
        feature_dim=contract["latent_dim"],
        frame_stack=contract["frame_stack"],
        action_dim=contract["action_dim"],
        hidden_dim=contract["hidden_dim"],
        action_chunk_size=contract["action_chunk_size"],
    ).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    return StateBCComponents(policy=policy, contract=contract, stats=stats)


def make_state_bc_evaluation_agent(
    checkpoint=None,
    stats_path=None,
    components=None,
    device="auto",
    execution_mode="open-loop",
    replan_interval=1,
    temporal_ensemble_decay=0.01,
):
    components = components or load_state_bc_components(checkpoint, stats_path, device)
    resolved_device = next(components.policy.parameters()).device

    def predict_chunk(stacked, deterministic):
        return components.policy(stacked)[0]

    return StateChunkAgent(
        agent_type="bc-state",
        encoder=None,
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
            "observation_space": "state",
            # The oracle never reads pixels. It still carries the resolution of
            # the dataset its states were recorded alongside, so it stays inside
            # the one evaluation protocol rather than needing an exemption.
            "training_observation_resolution": bc_training_observation_resolution(
                components.stats
            ),
        },
    )
