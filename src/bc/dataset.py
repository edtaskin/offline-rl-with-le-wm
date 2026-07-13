import torch
import numpy as np
from torch.utils.data import Dataset


PUSHT_ACTION_LOW = torch.tensor([-1.0, -1.0], dtype=torch.float32)
PUSHT_ACTION_HIGH = torch.tensor([1.0, 1.0], dtype=torch.float32)
PUSHT_ACTION_SCALE = 100.0
LEWM_IMAGE_SIZE = (224, 224)
LEWM_IMAGE_MEAN = [0.485, 0.456, 0.406]
LEWM_IMAGE_STD = [0.229, 0.224, 0.225]
LEWM_IMAGE_NORMALIZATION = "imagenet"


def absolute_to_relative_action(action, agent_position, action_scale=PUSHT_ACTION_SCALE):
    """Convert absolute PushT target pixels to SWM PushT relative controls."""
    relative_action = (action - agent_position) / action_scale
    action_low = PUSHT_ACTION_LOW.to(relative_action.device)
    action_high = PUSHT_ACTION_HIGH.to(relative_action.device)
    return torch.clamp(relative_action, action_low, action_high)


class PushTLeWMDataset(Dataset):
    def __init__(self, data_path, frame_stack=5, frame_stride=1, action_chunk_size=5):
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")

        data = np.load(data_path, allow_pickle=True)
        
        # Load images and ensure (N, C, H, W) format
        raw_images = data['images']
        if raw_images.shape[-1] == 3:
            raw_images = np.transpose(raw_images, (0, 3, 1, 2))
            
        self.images = torch.tensor(raw_images, dtype=torch.float32) / 255.0
        raw_actions = torch.tensor(data['actions'], dtype=torch.float32)
        raw_states = torch.tensor(data['states'], dtype=torch.float32)
        
        self.episode_ends = data['episode_ends']
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        self.action_chunk_size = action_chunk_size

        if len(self.images) != len(raw_actions):
            raise ValueError(f"images/actions length mismatch: {len(self.images)} vs {len(raw_actions)}")
        if len(self.images) != len(raw_states):
            raise ValueError(f"images/states length mismatch: {len(self.images)} vs {len(raw_states)}")
        if raw_actions.shape[-1] != 2:
            raise ValueError(f"expected 2D PushT actions, got shape {tuple(raw_actions.shape)}")
        if raw_states.shape[-1] < 2:
            raise ValueError(f"expected PushT states with agent x/y, got shape {tuple(raw_states.shape)}")
        if len(self.episode_ends) == 0 or int(self.episode_ends[-1]) != len(self.images):
            raise ValueError("episode_ends must be non-empty and end at the dataset length")
        
        # Match the preprocessing used by the official LeWM evaluation path.
        try:
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError(
                "torchvision is required to build image-based LeWM observations. "
                "Install torchvision, use an existing latent cache, or run with the intended project environment."
            ) from exc
        self.image_transform = T.Compose(
            [
                T.Resize(LEWM_IMAGE_SIZE, antialias=True),
                T.Normalize(mean=LEWM_IMAGE_MEAN, std=LEWM_IMAGE_STD),
            ]
        )

        action_min = PUSHT_ACTION_LOW.clone()
        action_max = PUSHT_ACTION_HIGH.clone()
        
        self.stats = {
            'action_min': action_min,
            'action_max': action_max,
            'action_space': 'swm_relative',
            'action_scale': PUSHT_ACTION_SCALE,
            'frame_stride': frame_stride,
            'action_chunk_size': action_chunk_size,
            'image_size': LEWM_IMAGE_SIZE,
            'image_normalization': LEWM_IMAGE_NORMALIZATION,
            'image_mean': LEWM_IMAGE_MEAN,
            'image_std': LEWM_IMAGE_STD,
        }
        
        # The diffusion-policy data stores absolute pixel targets. SWM PushT expects
        # relative controls: env target = agent_xy + action * action_scale.
        self.actions = absolute_to_relative_action(raw_actions, raw_states[:, :2])

        # Precompute episode boundaries for safe frame stacking
        self.ep_starts = np.zeros_like(self.episode_ends)
        self.ep_starts[1:] = self.episode_ends[:-1]

    def __len__(self):
        return len(self.images)

    def _get_frame_indices(self, idx, ep_start):
        frame_indices = []
        for i in range(self.frame_stack - 1, -1, -1):
            frame_idx = max(ep_start, idx - i * self.frame_stride)
            frame_indices.append(frame_idx)
        return frame_indices

    def __getitem__(self, idx):
        # Find which episode this index belongs to
        ep_idx = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = self.ep_starts[ep_idx]
        
        # Build the dilated frame history without crossing episode boundaries.
        frame_indices = self._get_frame_indices(idx, ep_start)
            
        # Extract and stack images: Shape (FrameStack, C, H, W)
        obs_seq = self.images[frame_indices]
        obs_seq = self.image_transform(obs_seq)
        
        # Future action chunk, padded at episode boundaries without crossing into
        # the next demonstration.
        action_indices = []
        ep_end = self.episode_ends[ep_idx]
        for offset in range(self.action_chunk_size):
            action_idx = min(ep_end - 1, idx + offset)
            action_indices.append(action_idx)
        action_chunk = self.actions[action_indices]
        
        return obs_seq, action_chunk


class PushTLeWMLatentDataset(Dataset):
    def __init__(self, data_path, latent_cache_path, frame_stack=5, frame_stride=1, action_chunk_size=5):
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")

        data = np.load(data_path, allow_pickle=True)
        raw_actions = torch.tensor(data['actions'], dtype=torch.float32)
        raw_states = torch.tensor(data['states'], dtype=torch.float32)
        self.episode_ends = data['episode_ends']
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        self.action_chunk_size = action_chunk_size

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
        if raw_actions.shape[-1] != 2:
            raise ValueError(f"expected 2D PushT actions, got shape {tuple(raw_actions.shape)}")
        if raw_states.shape[-1] < 2:
            raise ValueError(f"expected PushT states with agent x/y, got shape {tuple(raw_states.shape)}")
        if len(self.episode_ends) == 0 or int(self.episode_ends[-1]) != len(self.latents):
            raise ValueError("episode_ends must be non-empty and end at the dataset length")

        action_min = PUSHT_ACTION_LOW.clone()
        action_max = PUSHT_ACTION_HIGH.clone()
        latent_dim = int(self.latents.shape[-1])
        self.stats = {
            'action_min': action_min,
            'action_max': action_max,
            'action_space': 'swm_relative',
            'action_scale': PUSHT_ACTION_SCALE,
            'frame_stride': frame_stride,
            'action_chunk_size': action_chunk_size,
            'image_size': LEWM_IMAGE_SIZE,
            'image_normalization': LEWM_IMAGE_NORMALIZATION,
            'image_mean': LEWM_IMAGE_MEAN,
            'image_std': LEWM_IMAGE_STD,
            'latent_cache_path': str(latent_cache_path),
            'latent_cache_metadata': self.latent_cache_metadata,
            'latent_dim': latent_dim,
        }

        self.actions = absolute_to_relative_action(raw_actions, raw_states[:, :2])
        self.ep_starts = np.zeros_like(self.episode_ends)
        self.ep_starts[1:] = self.episode_ends[:-1]

    def __len__(self):
        return len(self.latents)

    def _get_frame_indices(self, idx, ep_start):
        frame_indices = []
        for i in range(self.frame_stack - 1, -1, -1):
            frame_idx = max(ep_start, idx - i * self.frame_stride)
            frame_indices.append(frame_idx)
        return frame_indices

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = self.ep_starts[ep_idx]
        frame_indices = self._get_frame_indices(idx, ep_start)
        stacked_latents = self.latents[frame_indices]

        action_indices = []
        ep_end = self.episode_ends[ep_idx]
        for offset in range(self.action_chunk_size):
            action_idx = min(ep_end - 1, idx + offset)
            action_indices.append(action_idx)
        action_chunk = self.actions[action_indices]

        return stacked_latents, action_chunk
