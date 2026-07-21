import numpy as np
import torch
from torch.utils.data import Dataset

from src.bc.history import action_chunk_indices, history_indices


PUSHT_ACTION_LOW = torch.tensor([-1.0, -1.0], dtype=torch.float32)
PUSHT_ACTION_HIGH = torch.tensor([1.0, 1.0], dtype=torch.float32)
PUSHT_ACTION_SCALE = 100.0


def absolute_to_relative_action(action, agent_position, action_scale=PUSHT_ACTION_SCALE):
    """Convert absolute PushT target pixels to SWM PushT relative controls."""
    relative_action = (action - agent_position) / action_scale
    action_low = PUSHT_ACTION_LOW.to(relative_action.device)
    action_high = PUSHT_ACTION_HIGH.to(relative_action.device)
    return torch.clamp(relative_action, action_low, action_high)


class _PushTTemporalDataset(Dataset):
    def _initialize_temporal_contract(
        self,
        raw_actions,
        raw_states,
        episode_ends,
        frame_stack,
        frame_stride,
        action_chunk_size,
    ):
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")
        if raw_actions.shape[-1] != 2:
            raise ValueError(f"expected 2D PushT actions, got shape {tuple(raw_actions.shape)}")
        if raw_states.shape[-1] < 2:
            raise ValueError(f"expected PushT states with agent x/y, got shape {tuple(raw_states.shape)}")

        self.episode_ends = np.asarray(episode_ends)
        if len(self.episode_ends) == 0 or int(self.episode_ends[-1]) != len(raw_actions):
            raise ValueError("episode_ends must be non-empty and end at the dataset length")
        self.episode_starts = np.zeros_like(self.episode_ends)
        self.episode_starts[1:] = self.episode_ends[:-1]
        self.ep_starts = self.episode_starts
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        self.action_chunk_size = action_chunk_size
        self.actions = absolute_to_relative_action(raw_actions, raw_states[:, :2])
        self.stats = {
            "action_min": PUSHT_ACTION_LOW.clone(),
            "action_max": PUSHT_ACTION_HIGH.clone(),
            "action_space": "swm_relative",
            "action_scale": PUSHT_ACTION_SCALE,
            "frame_stride": frame_stride,
            "action_chunk_size": action_chunk_size,
        }

    def _episode_index(self, index):
        return int(np.searchsorted(self.episode_ends, index, side="right"))

    def _get_frame_indices(self, index, episode_start):
        return history_indices(index, episode_start, self.frame_stack, self.frame_stride)

    def _get_action_indices(self, index, episode_end):
        return action_chunk_indices(index, episode_end, self.action_chunk_size)

    def _sample_indices(self, index):
        episode_index = self._episode_index(index)
        frame_indices = self._get_frame_indices(index, int(self.episode_starts[episode_index]))
        action_indices = self._get_action_indices(index, int(self.episode_ends[episode_index]))
        return frame_indices, action_indices


class PushTImageDataset(_PushTTemporalDataset):
    def __init__(self, data_path, frame_stack=5, frame_stride=1, action_chunk_size=5):
        super().__init__()
        with np.load(data_path, allow_pickle=True) as data:
            raw_images = np.asarray(data["images"])
            raw_actions = torch.tensor(data["actions"], dtype=torch.float32)
            raw_states = torch.tensor(data["states"], dtype=torch.float32)
            episode_ends = np.asarray(data["episode_ends"])
        if raw_images.ndim != 4:
            raise ValueError(
                f"expected NHWC or NCHW RGB images, got shape {tuple(raw_images.shape)}"
            )
        if raw_images.shape[-1] == 3:
            raw_images = np.transpose(raw_images, (0, 3, 1, 2))
        elif raw_images.shape[1] != 3:
            raise ValueError(
                f"expected NHWC or NCHW RGB images, got shape {tuple(raw_images.shape)}"
            )
        # Preserve the dataset's storage dtype and share the NumPy allocation.
        # In particular, converting a 224x224 uint8 dataset to float32 here would
        # require roughly 15 GB before the one-time LeWM latent-cache pass. The
        # shared LeWM preprocessor accepts both uint8 [0, 255] and float images.
        self.images = torch.from_numpy(raw_images)
        if len(self.images) != len(raw_actions):
            raise ValueError(f"images/actions length mismatch: {len(self.images)} vs {len(raw_actions)}")
        if len(self.images) != len(raw_states):
            raise ValueError(f"images/states length mismatch: {len(self.images)} vs {len(raw_states)}")
        self._initialize_temporal_contract(
            raw_actions,
            raw_states,
            episode_ends,
            frame_stack,
            frame_stride,
            action_chunk_size,
        )
        self.stats.update(
            {
                "source_image_shape": list(self.images.shape[-2:]),
                "source_image_dtype": str(self.images.dtype),
            }
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        frame_indices, action_indices = self._sample_indices(index)
        return self.images[frame_indices], self.actions[action_indices]


class PushTLatentDataset(_PushTTemporalDataset):
    def __init__(self, data_path, latent_cache_path, frame_stack=5, frame_stride=1, action_chunk_size=5):
        super().__init__()
        with np.load(data_path, allow_pickle=True) as data:
            raw_actions = torch.tensor(data["actions"], dtype=torch.float32)
            raw_states = torch.tensor(data["states"], dtype=torch.float32)
            episode_ends = np.asarray(data["episode_ends"])
        cache_payload = torch.load(latent_cache_path, map_location="cpu")
        if isinstance(cache_payload, dict) and "latents" in cache_payload:
            self.latents = cache_payload["latents"].float()
            self.latent_cache_metadata = cache_payload.get("metadata", {})
        elif torch.is_tensor(cache_payload):
            self.latents = cache_payload.float()
            self.latent_cache_metadata = {}
        else:
            raise ValueError(f"invalid latent cache payload in {latent_cache_path}")
        if self.latents.ndim != 2:
            raise ValueError(f"expected cached latents with shape (N, D), got {tuple(self.latents.shape)}")
        if len(self.latents) != len(raw_actions):
            raise ValueError(f"latents/actions length mismatch: {len(self.latents)} vs {len(raw_actions)}")
        if len(self.latents) != len(raw_states):
            raise ValueError(f"latents/states length mismatch: {len(self.latents)} vs {len(raw_states)}")
        self._initialize_temporal_contract(
            raw_actions,
            raw_states,
            episode_ends,
            frame_stack,
            frame_stride,
            action_chunk_size,
        )
        self.stats.update(
            {
                "latent_cache_path": str(latent_cache_path),
                "latent_cache_metadata": self.latent_cache_metadata,
                "latent_dim": int(self.latents.shape[-1]),
            }
        )

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        frame_indices, action_indices = self._sample_indices(index)
        return self.latents[frame_indices], self.actions[action_indices]
