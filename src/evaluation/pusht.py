"""Agent-independent evaluation loop for the PushT environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import gymnasium as gym
import numpy as np

from src.envs import PUSHT_FIXED_TARGET_POSE, PUSHT_RENDER_SHAPE, make_pusht_env
from src.evaluation.video import write_episode_video


FINAL_METRIC_KEYS = (
    "block_state_dist",
    "block_pos_dist",
    "block_angle_dist",
    "agent_block_dist",
    "block_goal_dist",
)


class EvaluationAgent(Protocol):
    agent_type: str
    metadata: dict[str, Any]

    def reset(self, seed: int) -> None: ...

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class PushTEvalConfig:
    env_id: str = "swm/PushT-v1"
    episodes: int = 20
    seed: int = 42
    max_episode_steps: int = 300
    observation_resolution: int = PUSHT_RENDER_SHAPE[0]
    fixed_target_pose: tuple[float, float, float] = tuple(PUSHT_FIXED_TARGET_POSE.tolist())
    fixed_target_block_success: bool = True
    fixed_target_max_reset_attempts: int = 100
    agent_block_coef: float = 0.0
    block_start_radius: float | None = None
    record_video: bool = False
    video_dir: str = "runs/eval_videos"
    video_fps: int = 10
    video_resolution: int = 512
    capture_traces: bool = False

    def validate(self):
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        if self.observation_resolution < 1:
            raise ValueError("observation_resolution must be positive")
        if self.block_start_radius is not None and self.block_start_radius < 0:
            raise ValueError("block_start_radius must be non-negative")
        if self.record_video and self.video_fps <= 0:
            raise ValueError("video_fps must be positive")


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    episode_return: float
    length: int
    success: float
    terminated: bool
    truncated: bool
    final_metrics: dict[str, float] = field(default_factory=dict)
    actions: list[list[float]] | None = None
    rewards: list[float] | None = None
    video_path: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class EvaluationResult:
    config: PushTEvalConfig
    agent_type: str
    agent_metadata: dict[str, Any]
    episodes: list[EpisodeResult]
    summary: dict[str, float]

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "agent_type": self.agent_type,
            "agent_metadata": self.agent_metadata,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "summary": self.summary,
        }


@dataclass
class RepeatedEvaluationResult:
    """Aggregate result for multiple evaluations with distinct seed ranges."""

    results: list[EvaluationResult]
    summary: dict[str, float | int]

    @property
    def config(self):
        return self.results[0].config

    @property
    def agent_type(self):
        return self.results[0].agent_type

    @property
    def agent_metadata(self):
        return self.results[0].agent_metadata

    @property
    def repeat_seeds(self):
        return [result.config.seed for result in self.results]

    @property
    def episodes(self):
        return [episode for result in self.results for episode in result.episodes]

    def to_dict(self):
        config = asdict(self.config)
        config.update(
            {
                "repeats": len(self.results),
                "repeat_seeds": list(self.repeat_seeds),
            }
        )
        episodes = []
        repeat_summaries = []
        for repeat, (seed, result) in enumerate(zip(self.repeat_seeds, self.results)):
            repeat_summaries.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    **result.summary,
                }
            )
            episodes.extend(
                {"repeat": repeat, **episode.to_dict()} for episode in result.episodes
            )
        return {
            "config": config,
            "agent_type": self.agent_type,
            "agent_metadata": self.agent_metadata,
            "repeat_summaries": repeat_summaries,
            "episodes": episodes,
            "summary": self.summary,
        }


def success_from_info(info, terminated):
    for key in ("success", "is_success", "task_success", "block_success"):
        if key in info:
            return float(info[key])
    return float(terminated)


def make_evaluation_env(config: PushTEvalConfig):
    config.validate()
    env = make_pusht_env(
        env_id=config.env_id,
        max_episode_steps=config.max_episode_steps,
        align_sampled_goal_to_fixed_target=True,
        fixed_target_pose=np.asarray(config.fixed_target_pose, dtype=float),
        fixed_target_block_success=config.fixed_target_block_success,
        fixed_target_max_reset_attempts=config.fixed_target_max_reset_attempts,
        fixed_target_agent_block_coef=config.agent_block_coef,
        block_start_near_goal=config.block_start_radius is not None,
        block_start_radius=config.block_start_radius or 0.0,
        resolution=config.observation_resolution,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    return env


def run_episode(env, agent: EvaluationAgent, config: PushTEvalConfig, episode_index: int):
    episode_seed = config.seed + episode_index
    observation, info = env.reset(seed=episode_seed)
    agent.reset(episode_seed)
    frames = [np.asarray(observation).copy()] if config.record_video else None
    actions = [] if config.capture_traces else None
    rewards = [] if config.capture_traces else None
    episode_return = 0.0
    length = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.asarray(agent.act(observation, info), dtype=np.float32)
        if action.shape != env.action_space.shape:
            raise ValueError(
                f"agent returned action shape {action.shape}; expected {env.action_space.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("agent returned a non-finite action")
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        length += 1
        if frames is not None:
            frames.append(np.asarray(observation).copy())
        if actions is not None:
            actions.append(action.tolist())
            rewards.append(float(reward))

    success = success_from_info(info, terminated)
    final_metrics = {}
    for key in FINAL_METRIC_KEYS:
        if key in info and np.isscalar(info[key]):
            final_metrics[key] = float(info[key])
    result = EpisodeResult(
        episode=episode_index,
        seed=episode_seed,
        episode_return=episode_return,
        length=length,
        success=success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        final_metrics=final_metrics,
        actions=actions,
        rewards=rewards,
    )
    if frames is not None:
        video_path = write_episode_video(
            frames,
            config.video_dir,
            episode_index,
            bool(success),
            fps=config.video_fps,
            resolution=config.video_resolution,
        )
        result.video_path = str(video_path) if video_path is not None else None
    return result


def summarize_results(episodes):
    returns = np.asarray([episode.episode_return for episode in episodes], dtype=float)
    lengths = np.asarray([episode.length for episode in episodes], dtype=float)
    successes = np.asarray([episode.success for episode in episodes], dtype=float)
    summary = {
        "episodes": int(len(episodes)),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(lengths.mean()),
        "success_rate": float(successes.mean()),
        "terminated_rate": float(np.mean([episode.terminated for episode in episodes])),
        "truncated_rate": float(np.mean([episode.truncated for episode in episodes])),
    }
    for key in FINAL_METRIC_KEYS:
        values = [episode.final_metrics[key] for episode in episodes if key in episode.final_metrics]
        if values:
            summary[f"mean_final_{key}"] = float(np.mean(values))
    return summary


def aggregate_evaluation_results(results):
    """Pool equally sized repeated evaluations into one reliable estimate."""

    results = list(results)
    if not results:
        raise ValueError("at least one evaluation result is required")
    episode_counts = {len(result.episodes) for result in results}
    if len(episode_counts) != 1:
        raise ValueError("all repeated evaluations must have the same episode count")
    first = results[0]
    if any(result.agent_type != first.agent_type for result in results[1:]):
        raise ValueError("all repeated evaluations must use the same agent type")

    episodes = [episode for result in results for episode in result.episodes]
    summary = summarize_results(episodes)
    summary.update(
        {
            "repeats": len(results),
            "episodes_per_repeat": episode_counts.pop(),
        }
    )
    return RepeatedEvaluationResult(
        results=results,
        summary=summary,
    )


def make_repeat_seeds(seed, repeats, episodes):
    """Derive deterministic, non-overlapping episode-seed ranges."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    return [int(seed) + repeat * int(episodes) for repeat in range(int(repeats))]


def run_repeated_evaluation(
    agent: EvaluationAgent,
    config: PushTEvalConfig,
    *,
    repeats=3,
    env_factory: Callable[[PushTEvalConfig], gym.Env] | None = None,
):
    """Run and pool repeated evaluations using one canonical seed protocol.

    ``env_factory`` lets callers supply an environment variant while retaining
    the exact same repeat construction, episode loop, and aggregation.
    """

    config.validate()
    repeat_seeds = make_repeat_seeds(config.seed, repeats, config.episodes)
    results = []
    for repeat, repeat_seed in enumerate(repeat_seeds):
        video_dir = (
            Path(config.video_dir)
            / f"repeat_{repeat:02d}_seed_{repeat_seed}"
        )
        repeat_config = replace(
            config,
            seed=repeat_seed,
            video_dir=str(video_dir),
        )
        print(
            f"Repeat {repeat + 1}/{repeats} | agent={agent.agent_type} | "
            f"fixed-target episodes={repeat_config.episodes} | "
            f"seeds={repeat_seed}..{repeat_seed + repeat_config.episodes - 1}"
        )
        env = env_factory(repeat_config) if env_factory is not None else None
        try:
            results.append(run_evaluation(agent, repeat_config, env=env))
        finally:
            if env is not None:
                env.close()
    return aggregate_evaluation_results(results)


def run_evaluation(agent: EvaluationAgent, config: PushTEvalConfig, env=None):
    config.validate()
    owns_env = env is None
    env = make_evaluation_env(config) if env is None else env
    episode_results = []
    try:
        for episode_index in range(config.episodes):
            result = run_episode(env, agent, config, episode_index)
            episode_results.append(result)
            print(
                f"  episode {episode_index:03d}: success={result.success:.0f} "
                f"return={result.episode_return:8.1f} len={result.length:5d}"
            )
    finally:
        if owns_env:
            env.close()
    summary = summarize_results(episode_results)
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return EvaluationResult(
        config=config,
        agent_type=agent.agent_type,
        agent_metadata=dict(agent.metadata),
        episodes=episode_results,
        summary=summary,
    )
