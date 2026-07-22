"""Paired closed-loop evaluation for ablation-local BC heads."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from src.envs import make_pusht_env
from src.evaluation.agents import LatentChunkAgent, resolve_device
from src.evaluation.pusht import (
    PushTEvalConfig,
    make_repeat_seeds,
    run_repeated_evaluation,
)

from .cache import path_fingerprint
from .encoders import load_encoder
from .shifts import PostRenderShiftWrapper, get_condition
from .training import load_policy


def config_for_condition(config, condition_name):
    condition = get_condition(condition_name)
    resolution = condition.effective_resolution(config.observation_resolution)
    return replace(config, observation_resolution=resolution)


def make_shifted_evaluation_env(config, condition_name):
    config = config_for_condition(config, condition_name)
    config.validate()
    condition = get_condition(condition_name)
    kwargs = {}
    init_value = condition.renderer_init_value()
    if init_value is not None:
        kwargs["init_value"] = init_value
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
        **kwargs,
    )
    if condition.has_post_render_shift:
        env = PostRenderShiftWrapper(env, condition)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    return env


def save_environment_screenshots(
    *,
    output_root,
    condition_names,
    seeds=(1000, 1001),
    block_start_radius=200.0,
    max_episode_steps=300,
    observation_resolution=96,
):
    condition_names = list(condition_names)
    seeds = [int(seed) for seed in seeds]
    if not condition_names or not seeds:
        raise ValueError("at least one condition and screenshot seed are required")
    screenshot_root = Path(output_root) / "environment_screenshots"
    screenshot_root.mkdir(parents=True, exist_ok=True)
    frames = {}
    condition_resolutions = {}
    config = PushTEvalConfig(
        episodes=1,
        seed=seeds[0],
        max_episode_steps=int(max_episode_steps),
        observation_resolution=int(observation_resolution),
        block_start_radius=float(block_start_radius),
    )
    for condition_name in condition_names:
        condition_config = config_for_condition(config, condition_name)
        condition_resolutions[condition_name] = condition_config.observation_resolution
        env = make_shifted_evaluation_env(condition_config, condition_name)
        condition_dir = screenshot_root / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        try:
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                frame = np.asarray(observation, dtype=np.uint8).copy()
                frames[(condition_name, seed)] = frame
                Image.fromarray(frame).save(condition_dir / f"seed_{seed}.png")
        finally:
            env.close()
    frame_height = frame_width = int(observation_resolution)
    label_width = 150
    header_height = 24
    grid = Image.new(
        "RGB",
        (
            label_width + len(seeds) * frame_width,
            header_height + len(condition_names) * frame_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for column, seed in enumerate(seeds):
        draw.text(
            (label_width + column * frame_width + 4, 5),
            f"seed {seed}",
            fill="black",
        )
    for row, condition_name in enumerate(condition_names):
        y = header_height + row * frame_height
        draw.text((5, y + frame_height // 2 - 6), condition_name, fill="black")
        for column, seed in enumerate(seeds):
            frame = Image.fromarray(frames[(condition_name, seed)])
            if frame.size != (frame_width, frame_height):
                frame = frame.resize(
                    (frame_width, frame_height), Image.Resampling.LANCZOS
                )
            grid.paste(
                frame,
                (label_width + column * frame_width, y),
            )
    grid_path = screenshot_root / "evaluation_conditions.png"
    grid.save(grid_path)
    manifest = {
        "conditions": condition_names,
        "seeds": seeds,
        "block_start_radius": float(block_start_radius),
        "max_episode_steps": int(max_episode_steps),
        "observation_resolution": int(observation_resolution),
        "condition_resolutions": condition_resolutions,
        "paired_physical_initial_states": True,
        "contact_sheet": str(grid_path),
    }
    manifest_path = screenshot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "root": screenshot_root,
        "contact_sheet": grid_path,
        "manifest": manifest_path,
        "images": [
            screenshot_root / condition / f"seed_{seed}.png"
            for condition in condition_names
            for seed in seeds
        ],
    }


def make_ablation_agent(checkpoint_path, encoder, device):
    policy, metadata = load_policy(checkpoint_path, device)
    contract = metadata["contract"]
    cached_encoder = metadata.get("cache_metadata", {}).get("encoder", {})
    if int(cached_encoder.get("feature_dim", -1)) != int(encoder.feature_dim):
        raise RuntimeError("evaluation encoder feature dimension does not match training")
    expected_checkpoint = cached_encoder.get("checkpoint")
    if expected_checkpoint and path_fingerprint(encoder.checkpoint_path) != expected_checkpoint:
        raise RuntimeError(
            "evaluation encoder checkpoint does not match the checkpoint used to build "
            "the training cache"
        )

    def predict_chunk(stacked, _deterministic):
        return policy(stacked)[0]

    return LatentChunkAgent(
        agent_type=f"bc-{metadata['encoder']}",
        encoder=encoder,
        predict_chunk=predict_chunk,
        contract=contract,
        device=device,
        deterministic=True,
        execution_mode="open-loop",
        metadata={
            "ablation": "lewm_visual_robustness",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "encoder": metadata["encoder"],
            "train_condition": metadata["train_condition"],
            "training_seed": metadata["seed"],
        },
    ), metadata


def evaluation_path(output_root, metadata, condition_name):
    return (
        Path(output_root)
        / "evaluations"
        / metadata["encoder"]
        / f"train_{metadata['train_condition']}"
        / f"seed_{metadata['seed']}"
        / f"eval_{condition_name}.json"
    )


def _existing_evaluation_matches(
    path, *, metadata, condition_name, config, repeats
):
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    ablation = payload.get("ablation", {})
    actual_config = payload.get("config", {})
    expected = {
        "encoder": metadata["encoder"],
        "train_condition": metadata["train_condition"],
        "eval_condition": condition_name,
        "training_seed": metadata["seed"],
    }
    expected_config = {
        "episodes": config.episodes,
        "seed": config.seed,
        "repeats": int(repeats),
        "repeat_seeds": make_repeat_seeds(
            config.seed, repeats, config.episodes
        ),
        "max_episode_steps": config.max_episode_steps,
        "observation_resolution": config.observation_resolution,
        "block_start_radius": config.block_start_radius,
    }
    expected_episode_count = int(config.episodes) * int(repeats)
    return (
        all(ablation.get(key) == value for key, value in expected.items())
        and all(
            actual_config.get(key) == value
            for key, value in expected_config.items()
        )
        and len(payload.get("repeat_summaries", [])) == int(repeats)
        and len(payload.get("episodes", [])) == expected_episode_count
    )


def evaluate_checkpoint(
    *,
    checkpoint_path,
    output_root,
    condition_names,
    device="auto",
    episodes=50,
    repeats=3,
    eval_seed=42,
    max_episode_steps=300,
    observation_resolution=96,
    block_start_radius=200.0,
    video=False,
    force=False,
    encoder=None,
):
    resolved_device = resolve_device(device)
    metadata = json.loads(Path(checkpoint_path).with_name("metadata.json").read_text())
    encoder = encoder or load_encoder(metadata["encoder"], resolved_device)
    agent, metadata = make_ablation_agent(checkpoint_path, encoder, resolved_device)
    paths = []
    for condition_name in condition_names:
        output_path = evaluation_path(output_root, metadata, condition_name)
        video_dir = output_path.parent / f"videos_{condition_name}"
        config = config_for_condition(
            PushTEvalConfig(
                episodes=int(episodes),
                seed=int(eval_seed),
                max_episode_steps=int(max_episode_steps),
                observation_resolution=int(observation_resolution),
                block_start_radius=float(block_start_radius),
                record_video=bool(video),
                video_dir=str(video_dir),
            ),
            condition_name,
        )
        if output_path.exists() and not force:
            if not _existing_evaluation_matches(
                output_path,
                metadata=metadata,
                condition_name=condition_name,
                config=config,
                repeats=repeats,
            ):
                raise RuntimeError(
                    f"existing evaluation is incompatible with the requested settings: "
                    f"{output_path}; use --force to replace it"
                )
            print(f"Using existing evaluation: {output_path}")
            paths.append(output_path)
            continue
        result = run_repeated_evaluation(
            agent,
            config,
            repeats=int(repeats),
            env_factory=lambda repeat_config: make_shifted_evaluation_env(
                repeat_config, condition_name
            ),
        )
        payload = result.to_dict()
        payload["ablation"] = {
            "name": "lewm_visual_robustness",
            "encoder": metadata["encoder"],
            "train_condition": metadata["train_condition"],
            "eval_condition": condition_name,
            "training_seed": metadata["seed"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths.append(output_path)
    return paths
