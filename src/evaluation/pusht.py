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

CANONICAL_V1 = "canonical_v1"
CANONICAL_V2 = "canonical_v2"
EVALUATION_PROTOCOLS = (CANONICAL_V1, CANONICAL_V2, "custom")

# canonical_v2 deliberately balances the Cartesian product of these bands.
# Translation is the geometric block-centroid-to-goal-centroid distance. Using
# centroids keeps this axis independent of rotation despite the T body's
# off-center pose origin; rotation is the wrapped absolute angular error.
CANONICAL_V2_DISTANCE_THRESHOLDS = (70.0, 140.0)
CANONICAL_V2_ANGLE_THRESHOLDS = (float(np.pi / 4),)
CANONICAL_V2_COMPLETION_BUDGETS = (50, 100, 200, 300)
_DISTANCE_BAND_NAMES = ("near", "mid", "far")
_ANGLE_BAND_NAMES = ("aligned", "misaligned")
CANONICAL_V2_STRATA = tuple(
    f"{distance}_{angle}"
    for distance in _DISTANCE_BAND_NAMES
    for angle in _ANGLE_BAND_NAMES
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
    protocol: str = "custom"
    # Canonical evaluation supplies an explicit, reproducible set of
    # well-separated episode seeds derived from ``seed``. Other callers may omit
    # it and retain the legacy arithmetic ``seed + i * seed_stride`` schedule.
    episode_seeds: tuple[int, ...] | None = None
    # canonical_v2 records the stratum assigned while constructing its fixed
    # seed suite. Keeping it beside the seeds makes the benchmark auditable and
    # lets run_episode verify that a reset still produces the expected state.
    episode_strata: tuple[str, ...] | None = None
    # Gap between consecutive episode seeds. Consecutive integer seeds collide in
    # the underlying PushT reset roughly 17% of the time, which silently turns
    # some evaluation episodes into duplicates of their neighbour. The near-goal
    # start wrapper hides this (it resamples the block from its own reseeded RNG),
    # so a stride is only needed for unrestricted starts; any value >= 7 was
    # measured to give fully distinct start states.
    seed_stride: int = 1
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
    allow_resolution_mismatch: bool = False
    visualize_starts: bool = False
    distance_thresholds: tuple[float, ...] = ()
    angle_thresholds: tuple[float, ...] = ()
    completion_budgets: tuple[int, ...] = ()

    def validate(self):
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.protocol not in EVALUATION_PROTOCOLS:
            raise ValueError(
                f"protocol must be one of {EVALUATION_PROTOCOLS}, got {self.protocol!r}"
            )
        if self.seed_stride < 1:
            raise ValueError("seed_stride must be at least 1")
        if self.episode_seeds is not None:
            if len(self.episode_seeds) != self.episodes:
                raise ValueError(
                    "episode_seeds must contain exactly one seed per episode"
                )
            seeds = [int(seed) for seed in self.episode_seeds]
            if any(seed < 0 for seed in seeds):
                raise ValueError("episode seeds must be non-negative")
            if len(set(seeds)) != len(seeds):
                raise ValueError("episode seeds must be unique")
        if self.episode_strata is not None:
            if len(self.episode_strata) != self.episodes:
                raise ValueError(
                    "episode_strata must contain exactly one label per episode"
                )
            unknown = set(self.episode_strata) - set(CANONICAL_V2_STRATA)
            if unknown:
                raise ValueError(f"unknown episode strata: {sorted(unknown)}")
        for name, thresholds in (
            ("distance_thresholds", self.distance_thresholds),
            ("angle_thresholds", self.angle_thresholds),
        ):
            values = tuple(float(value) for value in thresholds)
            if any(not np.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain finite positive values")
            if any(right <= left for left, right in zip(values, values[1:])):
                raise ValueError(f"{name} must be strictly increasing")
        budgets = tuple(int(value) for value in self.completion_budgets)
        if any(value < 1 for value in budgets):
            raise ValueError("completion_budgets must be positive")
        if any(right <= left for left, right in zip(budgets, budgets[1:])):
            raise ValueError("completion_budgets must be strictly increasing")
        if bool(self.distance_thresholds) != bool(self.angle_thresholds):
            raise ValueError(
                "distance_thresholds and angle_thresholds must be supplied together"
            )
        if self.distance_thresholds and (
            len(self.distance_thresholds) != 2 or len(self.angle_thresholds) != 1
        ):
            raise ValueError(
                "PushT stratification requires two distance thresholds and one angle threshold"
            )
        if self.protocol == CANONICAL_V2:
            if not self.fixed_target_block_success:
                raise ValueError(
                    "canonical_v2 requires fixed_target_block_success=True"
                )
            if self.block_start_radius is None or not np.isclose(
                self.block_start_radius, 200.0
            ):
                raise ValueError("canonical_v2 requires block_start_radius=200")
            if tuple(self.distance_thresholds) != CANONICAL_V2_DISTANCE_THRESHOLDS:
                raise ValueError(
                    "canonical_v2 requires its fixed distance thresholds"
                )
            if tuple(self.angle_thresholds) != CANONICAL_V2_ANGLE_THRESHOLDS:
                raise ValueError("canonical_v2 requires its fixed angle thresholds")
            if tuple(self.completion_budgets) != CANONICAL_V2_COMPLETION_BUDGETS:
                raise ValueError("canonical_v2 requires its fixed completion budgets")
            if self.max_episode_steps != CANONICAL_V2_COMPLETION_BUDGETS[-1]:
                raise ValueError("canonical_v2 requires max_episode_steps=300")
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
    initial_metrics: dict[str, float] = field(default_factory=dict)
    stratum: str | None = None
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
    summary: dict[str, float | int]
    strata: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "agent_type": self.agent_type,
            "agent_metadata": self.agent_metadata,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "summary": self.summary,
            "strata": self.strata,
        }


@dataclass
class RepeatedEvaluationResult:
    """Aggregate result for multiple evaluations with distinct seed ranges."""

    results: list[EvaluationResult]
    summary: dict[str, float | int]
    strata: list[dict[str, Any]] = field(default_factory=list)

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
            "strata": self.strata,
        }


@dataclass(frozen=True)
class StratifiedEpisodeSuite:
    """Fixed simulator reset seeds selected to fill canonical_v2 strata."""

    seeds: tuple[int, ...]
    strata: tuple[str, ...]
    candidates_examined: int
    counts: dict[str, int]


def success_from_info(info, terminated):
    for key in ("success", "is_success", "task_success", "block_success"):
        if key in info:
            return float(info[key])
    return float(terminated)


def scalar_metrics(info, keys=FINAL_METRIC_KEYS) -> dict[str, float]:
    """Copy finite scalar task metrics out of an environment info mapping."""

    metrics = {}
    for key in keys:
        if key in info and np.isscalar(info[key]):
            value = float(info[key])
            if np.isfinite(value):
                metrics[key] = value
    return metrics


def difficulty_stratum(
    initial_metrics: dict[str, float],
    distance_thresholds=CANONICAL_V2_DISTANCE_THRESHOLDS,
    angle_thresholds=CANONICAL_V2_ANGLE_THRESHOLDS,
) -> str:
    """Assign one start state to a translation x rotation difficulty cell."""

    try:
        distance = float(initial_metrics["block_goal_dist"])
        angle = float(initial_metrics["block_angle_dist"])
    except KeyError as exc:
        raise ValueError(
            "difficulty stratification requires block_goal_dist and block_angle_dist"
        ) from exc
    if not np.isfinite(distance) or not np.isfinite(angle):
        raise ValueError("difficulty metrics must be finite")

    distance_index = int(np.searchsorted(distance_thresholds, distance, side="right"))
    angle_index = int(np.searchsorted(angle_thresholds, angle, side="right"))
    if distance_index >= len(_DISTANCE_BAND_NAMES):
        raise ValueError("canonical_v2 supports exactly three distance bands")
    if angle_index >= len(_ANGLE_BAND_NAMES):
        raise ValueError("canonical_v2 supports exactly two angle bands")
    return f"{_DISTANCE_BAND_NAMES[distance_index]}_{_ANGLE_BAND_NAMES[angle_index]}"


def stratified_episode_quotas(episodes: int) -> dict[str, int]:
    """Allocate an equal deterministic quota across canonical_v2's six cells."""

    if episodes < len(CANONICAL_V2_STRATA):
        raise ValueError(
            f"canonical_v2 requires at least {len(CANONICAL_V2_STRATA)} episodes"
        )
    base, remainder = divmod(int(episodes), len(CANONICAL_V2_STRATA))
    return {
        stratum: base + int(index < remainder)
        for index, stratum in enumerate(CANONICAL_V2_STRATA)
    }


def make_evaluation_env(
    config: PushTEvalConfig,
    *,
    render_observations: bool = True,
    record_statistics: bool = True,
):
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
        render_obs=render_observations,
        resolution=config.observation_resolution,
    )
    if record_statistics:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    return env


def select_stratified_episode_seeds(
    config: PushTEvalConfig,
    candidate_seeds,
    *,
    env=None,
) -> StratifiedEpisodeSuite:
    """Select a deterministic fixed seed suite with equal difficulty quotas.

    Candidate reset seeds are inspected without taking actions. The accepted
    seeds can then be replayed for every checkpoint, so suite construction is
    independent of the evaluated policy.
    """

    config.validate()
    quotas = stratified_episode_quotas(config.episodes)
    accepted: list[int] = []
    labels: list[str] = []
    counts = {stratum: 0 for stratum in CANONICAL_V2_STRATA}
    seen: set[int] = set()
    owns_env = env is None
    if env is None:
        env = make_evaluation_env(
            config,
            render_observations=False,
            record_statistics=False,
        )

    examined = 0
    try:
        for candidate in candidate_seeds:
            seed = int(candidate)
            if seed < 0:
                raise ValueError("candidate seeds must be non-negative")
            if seed in seen:
                continue
            seen.add(seed)
            examined += 1
            _, info = env.reset(seed=seed)
            # A benchmark episode must require at least one action. Near/aligned
            # remains an explicit easy cell, but already-solved resets are not
            # allowed to inflate it.
            if success_from_info(info, False) > 0.5:
                continue
            label = difficulty_stratum(
                scalar_metrics(info),
                config.distance_thresholds,
                config.angle_thresholds,
            )
            if counts[label] >= quotas[label]:
                continue
            accepted.append(seed)
            labels.append(label)
            counts[label] += 1
            if counts == quotas:
                break
    finally:
        if owns_env:
            env.close()

    if counts != quotas:
        missing = {
            label: quotas[label] - counts[label]
            for label in CANONICAL_V2_STRATA
            if counts[label] < quotas[label]
        }
        raise RuntimeError(
            "candidate seed pool could not fill canonical_v2 strata; "
            f"missing={missing}, examined={examined}"
        )
    return StratifiedEpisodeSuite(
        seeds=tuple(accepted),
        strata=tuple(labels),
        candidates_examined=examined,
        counts=counts,
    )


def run_episode(env, agent: EvaluationAgent, config: PushTEvalConfig, episode_index: int):
    if config.episode_seeds is None:
        episode_seed = config.seed + episode_index * config.seed_stride
    else:
        episode_seed = int(config.episode_seeds[episode_index])
    observation, info = env.reset(seed=episode_seed)
    initial_metrics = scalar_metrics(info)
    expected_stratum = (
        config.episode_strata[episode_index]
        if config.episode_strata is not None
        else None
    )
    stratum = None
    if config.distance_thresholds and config.angle_thresholds:
        stratum = difficulty_stratum(
            initial_metrics,
            config.distance_thresholds,
            config.angle_thresholds,
        )
    if expected_stratum is not None and stratum != expected_stratum:
        raise RuntimeError(
            "evaluation reset no longer matches its canonical_v2 stratum: "
            f"seed={episode_seed}, expected={expected_stratum}, observed={stratum}"
        )
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
    final_metrics = scalar_metrics(info)
    result = EpisodeResult(
        episode=episode_index,
        seed=episode_seed,
        episode_return=episode_return,
        length=length,
        success=success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        initial_metrics=initial_metrics,
        stratum=stratum,
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


def _success_by_budget(episodes, budget: int) -> float:
    """Fraction of episodes successfully completed within ``budget`` steps."""

    return float(
        np.mean(
            [episode.success > 0.5 and episode.length <= budget for episode in episodes]
        )
    )


def _success_auc(episodes, horizon: int) -> float:
    """Normalized area under the empirical success-by-step curve."""

    return float(
        np.mean(
            [
                max(0.0, horizon - episode.length) / horizon
                if episode.success > 0.5
                else 0.0
                for episode in episodes
            ]
        )
    )


def summarize_strata(episodes, completion_budgets=()) -> list[dict[str, Any]]:
    """Return detailed per-cell metrics without nesting them in scalar summary."""

    grouped = {
        label: [episode for episode in episodes if episode.stratum == label]
        for label in CANONICAL_V2_STRATA
    }
    rows = []
    for label in CANONICAL_V2_STRATA:
        group = grouped[label]
        if not group:
            continue
        successes = np.asarray([episode.success for episode in group], dtype=float)
        lengths = np.asarray([episode.length for episode in group], dtype=float)
        row: dict[str, Any] = {
            "stratum": label,
            "episodes": int(len(group)),
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "mean_length": float(lengths.mean()),
        }
        for metric in FINAL_METRIC_KEYS:
            values = [
                episode.initial_metrics[metric]
                for episode in group
                if metric in episode.initial_metrics
            ]
            if values:
                row[f"mean_initial_{metric}"] = float(np.mean(values))
        for budget in completion_budgets:
            row[f"success_by_{int(budget)}"] = _success_by_budget(group, int(budget))
        if completion_budgets:
            row["success_auc"] = _success_auc(group, int(completion_budgets[-1]))
        rows.append(row)
    return rows


def summarize_results(episodes, completion_budgets=()):
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
    for key in FINAL_METRIC_KEYS:
        values = [
            episode.initial_metrics[key]
            for episode in episodes
            if key in episode.initial_metrics
        ]
        if values:
            summary[f"mean_initial_{key}"] = float(np.mean(values))

    budgets = tuple(int(value) for value in completion_budgets)
    for budget in budgets:
        summary[f"success_by_{budget}"] = _success_by_budget(episodes, budget)
    if budgets:
        summary["success_auc"] = _success_auc(episodes, budgets[-1])

    strata = summarize_strata(episodes, budgets)
    if strata:
        rates = np.asarray([row["success_rate"] for row in strata], dtype=float)
        summary["balanced_success_rate"] = float(rates.mean())
        summary["worst_stratum_success_rate"] = float(rates.min())
        hard = next(
            (row for row in strata if row["stratum"] == "far_misaligned"),
            None,
        )
        if hard is not None:
            summary["hard_success_rate"] = float(hard["success_rate"])
        for budget in budgets:
            key = f"success_by_{budget}"
            balanced = float(np.mean([row[key] for row in strata]))
            summary[f"balanced_{key}"] = balanced
        if budgets:
            summary["balanced_success_auc"] = float(
                np.mean([row["success_auc"] for row in strata])
            )
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
    summary = summarize_results(episodes, first.config.completion_budgets)
    summary.update(
        {
            "repeats": len(results),
            "episodes_per_repeat": episode_counts.pop(),
        }
    )
    return RepeatedEvaluationResult(
        results=results,
        summary=summary,
        strata=summarize_strata(episodes, first.config.completion_budgets),
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
    training_resolution = agent.metadata.get("training_observation_resolution")
    if (
        training_resolution is not None
        and int(training_resolution) != int(config.observation_resolution)
        and not config.allow_resolution_mismatch
    ):
        raise ValueError(
            "evaluation observation resolution does not match model training: "
            f"model={int(training_resolution)}, evaluation={config.observation_resolution}. "
            "Use the training resolution, or explicitly allow the mismatch for a "
            "resolution-robustness experiment."
        )
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
    summary = summarize_results(episode_results, config.completion_budgets)
    strata = summarize_strata(episode_results, config.completion_budgets)
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if strata:
        print("Starting-stratum summary:")
        for row in strata:
            fields = [
                f"episodes={row['episodes']}",
                f"success_rate={row['success_rate']:.3f}",
                f"mean_length={row['mean_length']:.2f}",
            ]
            fields.extend(
                f"success_by_{int(budget)}={row[f'success_by_{int(budget)}']:.3f}"
                for budget in config.completion_budgets
            )
            if "success_auc" in row:
                fields.append(f"success_auc={row['success_auc']:.3f}")
            print(f"  {row['stratum']}: " + " | ".join(fields))
    return EvaluationResult(
        config=config,
        agent_type=agent.agent_type,
        agent_metadata=dict(agent.metadata),
        episodes=episode_results,
        summary=summary,
        strata=strata,
    )
