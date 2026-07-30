"""Image-observation PushT env builder and dilated latent-history buffer.

The frozen-encoder latent PPO agent consumes a *dilated* stack of per-frame
latents, exactly like the BC policy at train/eval time. This module provides:

* :func:`make_latent_env` -- a thunk that builds one ``swm/PushT-v1`` instance
  returning RGB frames (via :func:`src.envs.make_pusht_env`) plus episode-stat
  recording, matching the manual parallel-env style of the trainer in
  :mod:`src.ppo.ppo`.
* :class:`LatentHistory` -- a per-env ring buffer of step latents that reproduces
  the dilated frame selection from ``src/bc/run_eval.py`` (``build_stacked_latents``).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

from src.envs import PUSHT_FIXED_TARGET_POSE, PUSHT_RENDER_SHAPE, make_pusht_env
from src.evaluation.pusht import success_from_info
from src.representations.history import LatentHistory


def make_latent_env(
    *,
    env_id: str = "swm/PushT-v1",
    seed: int = 0,
    idx: int = 0,
    max_episode_steps: int = 300,
    observation_resolution: int = PUSHT_RENDER_SHAPE[0],
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
            resolution=observation_resolution,
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
