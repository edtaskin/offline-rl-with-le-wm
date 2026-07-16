import tempfile
from collections import deque

import numpy as np
import torch

from src.bc.history import FeatureHistory
from src.bc.lewm import load_stable_worldmodel
from src.bc.tracking import json_safe
from src.bc.video import combine_world_panel_videos, is_video_file_path


def _to_swm_state(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape[0] >= 7:
        return state[:7].copy()
    if state.shape[0] == 5:
        return np.concatenate([state, np.zeros(2, dtype=np.float64)])
    raise ValueError(f"expected PushT state with 5 or 7 values, got shape {state.shape}")


class PushTNPZWorldDataset:
    """Minimal dataset adapter for swm.World.evaluate(dataset=...)."""

    column_names = ("pixels", "state", "proprio", "action")

    def __init__(self, data_path, image_size):
        with np.load(data_path, allow_pickle=True) as data:
            self.images = np.asarray(data["images"])
            self.states = np.stack([_to_swm_state(state) for state in data["states"]])
            self.actions = np.asarray(data["actions"], dtype=np.float32)
            self.episode_ends = np.asarray(data["episode_ends"], dtype=np.int64)
        self.image_size = tuple(image_size)
        self.episode_starts = np.zeros_like(self.episode_ends)
        self.episode_starts[1:] = self.episode_ends[:-1]
        if self.images.shape[-1] != 3:
            raise ValueError(f"expected images with last channel RGB, got {self.images.shape}")
        if len(self.images) != len(self.states):
            raise ValueError("images/states length mismatch")
        if len(self.actions) != len(self.states):
            raise ValueError("actions/states length mismatch")

    def episode_length(self, episode_index):
        return int(self.episode_ends[episode_index] - self.episode_starts[episode_index])

    def _resize_images(self, images):
        if images.shape[1:3] == self.image_size:
            return images
        import cv2

        return np.stack(
            [
                cv2.resize(
                    image,
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for image in images
            ],
            axis=0,
        )

    def load_chunk(self, episode_indices, start_steps, end_steps):
        chunks = []
        for episode_index, start_step, end_step in zip(episode_indices, start_steps, end_steps):
            abs_start = int(self.episode_starts[int(episode_index)] + int(start_step))
            abs_end = int(self.episode_starts[int(episode_index)] + int(end_step))
            states = self.states[abs_start:abs_end]
            images = self._resize_images(self.images[abs_start:abs_end])
            actions = self.actions[abs_start:abs_end]
            proprio = np.concatenate([states[:, :2], states[:, -2:]], axis=-1)
            chunks.append(
                {
                    "pixels": torch.as_tensor(images).permute(0, 3, 1, 2),
                    "state": states.copy(),
                    "proprio": proprio,
                    "action": actions.copy(),
                }
            )
        return chunks


def sample_world_eval_starts(dataset, num_episodes, goal_offset_steps, seed):
    valid = []
    for episode_index in range(len(dataset.episode_ends)):
        max_start = dataset.episode_length(episode_index) - goal_offset_steps - 1
        for start_step in range(max_start + 1):
            valid.append((episode_index, start_step))
    if not valid:
        raise ValueError(f"No valid dataset starts for goal_offset_steps={goal_offset_steps}.")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(valid), size=num_episodes, replace=num_episodes > len(valid))
    episode_indices, start_steps = zip(*(valid[int(index)] for index in sampled))
    return list(episode_indices), list(start_steps)


def install_pusht_goal_pose_setter():
    from stable_worldmodel.envs.pusht.env import PushT

    def _set_goal_state_and_pose(self, goal_state):
        goal_state = _to_swm_state(goal_state)
        self._set_goal_state(goal_state)
        self.goal_pose = goal_state[2:5].copy()

    PushT._set_goal_state_and_pose = _set_goal_state_and_pose


class LatentBCWorldPolicy:
    def __init__(self, extractor, policy, frame_stack, frame_stride, device):
        self.extractor = extractor
        self.policy = policy
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        self.device = device
        self.env = None
        self.feature_histories = None
        self.action_buffers = None

    def set_env(self, env):
        self.env = env
        self.feature_histories = [
            FeatureHistory(self.frame_stack, self.frame_stride) for _ in range(env.num_envs)
        ]
        self.action_buffers = [deque() for _ in range(env.num_envs)]

    def get_action(self, info_dict, **kwargs):
        if self.env is None:
            raise RuntimeError("LatentBCWorldPolicy.set_env must be called before get_action")
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            for env_index, should_flush in enumerate(np.asarray(needs_flush).reshape(-1)):
                if should_flush:
                    self.feature_histories[env_index].clear()
                    self.action_buffers[env_index].clear()
        pixels = np.asarray(info_dict["pixels"])[:, -1]
        image_batch = torch.as_tensor(pixels, dtype=torch.float32, device=self.device)
        image_batch = image_batch.permute(0, 3, 1, 2) / 255.0
        features = self.extractor.encode(image_batch).detach().cpu()
        actions = []
        for env_index in range(self.env.num_envs):
            self.feature_histories[env_index].append(features[env_index])
            if not self.action_buffers[env_index]:
                stacked = self.feature_histories[env_index].stacked(self.device)
                with torch.no_grad():
                    action_chunk = self.policy(stacked).squeeze(0).cpu()
                self.action_buffers[env_index].extend(
                    torch.clamp(action_chunk, -1.0, 1.0).numpy()
                )
            actions.append(self.action_buffers[env_index].popleft())
        return np.asarray(actions, dtype=np.float32)


def evaluate_lewm_world(
    args,
    extractor,
    policy,
    frame_stack,
    frame_stride,
    wandb_run=None,
):
    install_pusht_goal_pose_setter()
    image_size = extractor.preprocessing_metadata["image_size"]
    dataset = PushTNPZWorldDataset(args.eval_data_path, image_size=image_size)
    episode_indices, start_steps = sample_world_eval_starts(
        dataset,
        args.episodes,
        args.goal_offset_steps,
        args.eval_seed,
    )
    print(
        "Using swm.World.evaluate(dataset=...) for PushT BC: "
        f"episodes={args.episodes}, goal_offset_steps={args.goal_offset_steps}, "
        f"eval_budget={args.lewm_eval_budget}, seed={args.eval_seed}."
    )
    print(f"Sampled dataset episode indices: {episode_indices}")
    print(f"Sampled dataset start steps: {start_steps}")
    combine_video = args.video_path is not None and is_video_file_path(args.video_path)
    with tempfile.TemporaryDirectory(prefix="pusht-world-video-") as temporary_video_dir:
        world_video_path = temporary_video_dir if combine_video else args.video_path
        if combine_video:
            print(
                "Writing swm.World per-env panel videos to a temporary directory "
                f"before combining into {args.video_path}."
            )
        swm = load_stable_worldmodel()
        world = swm.World(
            "swm/PushT-v1",
            num_envs=args.episodes,
            image_shape=image_size,
            max_episode_steps=2 * args.lewm_eval_budget,
        )
        world.set_policy(
            LatentBCWorldPolicy(
                extractor=extractor,
                policy=policy,
                frame_stack=frame_stack,
                frame_stride=frame_stride,
                device=extractor.device,
            )
        )
        try:
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=episode_indices,
                start_steps=start_steps,
                goal_offset=args.goal_offset_steps,
                eval_budget=args.lewm_eval_budget,
                callables=[
                    {"method": "_set_state", "args": {"state": {"value": "state"}}},
                    {
                        "method": "_set_goal_state_and_pose",
                        "args": {"goal_state": {"value": "goal_state"}},
                    },
                ],
                video=world_video_path,
            )
        finally:
            world.close()
        if combine_video:
            combine_world_panel_videos(world_video_path, args.video_path)
    print(f"swm.World metrics: {metrics}")
    world_success_rate = float(metrics.get("success_rate", 0.0))
    normalized_success_rate = world_success_rate / 100.0
    print(f"World success rate: {world_success_rate:.2f}%")
    print(f"Success rate: {normalized_success_rate:.4f}")
    if wandb_run is not None:
        wandb_run.summary["eval/world_success_rate_percent"] = world_success_rate
        wandb_run.summary["eval/success_rate"] = normalized_success_rate
        wandb_run.summary["eval/world_episode_successes"] = json_safe(
            metrics.get("episode_successes")
        )
        wandb_run.log(
            {
                "eval/world_success_rate_percent": world_success_rate,
                "eval/world_success_rate": normalized_success_rate,
                "eval/success_rate": normalized_success_rate,
            },
            step=args.episodes,
        )
    return metrics
