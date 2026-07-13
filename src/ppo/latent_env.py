"""Image-observation PushT env builder and dilated latent-history buffer.

The frozen-encoder latent PPO agent consumes a *dilated* stack of per-frame
latents, exactly like the BC policy at train/eval time. This module provides:

* :func:`make_latent_env` -- a thunk that builds one ``swm/PushT-v1`` instance
  returning RGB frames (via :func:`src.envs.make_pusht_env`) plus episode-stat
  recording, matching the manual parallel-env style of :mod:`src.ppo.ppo`.
* :class:`LatentHistory` -- a per-env ring buffer of step latents that reproduces
  the dilated frame selection from ``src/bc/run_eval.py`` (``build_stacked_latents``).
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import gymnasium as gym
import torch

from src.envs import PUSHT_FIXED_TARGET_POSE, make_pusht_env


def success_from_info(info: dict, terminated: bool) -> float:
    """Episode success flag, mirroring ``run_eval._success_from_info``."""
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return float(info[key])
    return float(terminated)


def make_latent_env(
    *,
    env_id: str = "swm/PushT-v1",
    seed: int = 0,
    idx: int = 0,
    max_episode_steps: int = 300,
    record_stats: bool = True,
    fixed_target: bool = False,
    fixed_target_pose=PUSHT_FIXED_TARGET_POSE,
    fixed_target_block_success: bool = True,
    fixed_target_max_reset_attempts: int = 100,
    agent_block_coef: float = 0.0,
    block_start_near_goal: bool = False,
    block_start_radius: float = 50.0,
    reward_mode: str = "dense",
) -> Callable[[], gym.Env]:
    """Return a thunk building one image-observation PushT env.

    Observations are ``uint8`` RGB frames ``(H, W, 3)``. With ``fixed_target``,
    each sampled task is rigidly aligned to a fixed block target pose and uses
    block-pose reward/success (see
    :class:`src.envs.PushTAlignSampledGoalToFixedTargetWrapper`); otherwise the
    native dense reward (``-distance``) and success are used.

    ``block_start_near_goal`` starts each episode with the block within
    ``block_start_radius`` pixels of the green T center. ``reward_mode="sparse"``
    replaces the dense reward with ``1.0`` on success and ``0.0`` otherwise.
    """

    def thunk() -> gym.Env:
        env = make_pusht_env(
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            align_sampled_goal_to_fixed_target=fixed_target,
            fixed_target_pose=fixed_target_pose,
            fixed_target_block_success=fixed_target_block_success,
            fixed_target_max_reset_attempts=fixed_target_max_reset_attempts,
            fixed_target_agent_block_coef=agent_block_coef,
            block_start_near_goal=block_start_near_goal,
            block_start_radius=block_start_radius,
            reward_mode=reward_mode,
        )
        if record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


class LatentHistory:
    """Per-env dilated history of step latents -> stacked ``[frame_stack, D]``.

    Stores one latent (shape ``[latent_dim]``) per environment step and selects
    ``frame_stack`` of them spaced ``frame_stride`` steps apart, ending at the
    most recent, padding with the oldest available latent early in an episode.
    This is the exact selection used by ``run_eval.build_stacked_latents``.
    """

    def __init__(self, frame_stack: int, frame_stride: int):
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        self.max_len = (frame_stack - 1) * frame_stride + 1
        self._buf: deque[torch.Tensor] = deque(maxlen=self.max_len)

    def clear(self) -> None:
        self._buf.clear()

    def append(self, latent: torch.Tensor) -> None:
        """Append one step latent (shape ``[latent_dim]``)."""
        self._buf.append(latent)

    def __len__(self) -> int:
        return len(self._buf)

    def stacked(self) -> torch.Tensor:
        """Return the dilated stack ``[frame_stack, latent_dim]``."""
        if not self._buf:
            raise RuntimeError("LatentHistory is empty; append a latent first")
        history = list(self._buf)
        oldest = history[0]
        selected = []
        for offset in range(self.frame_stack - 1, -1, -1):
            idx = len(history) - 1 - offset * self.frame_stride
            selected.append(history[idx] if idx >= 0 else oldest)
        return torch.stack(selected, dim=0)
