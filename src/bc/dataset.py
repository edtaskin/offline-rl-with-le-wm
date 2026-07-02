import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


PUSHT_ACTION_LOW = torch.tensor([-1.0, -1.0], dtype=torch.float32)
PUSHT_ACTION_HIGH = torch.tensor([1.0, 1.0], dtype=torch.float32)
PUSHT_ACTION_SCALE = 100.0


def absolute_to_relative_action(action, agent_position, action_scale=PUSHT_ACTION_SCALE):
    """Convert absolute PushT target pixels to SWM PushT relative controls."""
    relative_action = (action - agent_position) / action_scale
    action_low = PUSHT_ACTION_LOW.to(relative_action.device)
    action_high = PUSHT_ACTION_HIGH.to(relative_action.device)
    return torch.clamp(relative_action, action_low, action_high)


class PushTLeWMDataset(Dataset):
    def __init__(self, data_path, frame_stack=5, action_chunk_size=5):
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
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
        
        # Resize to match what LeWM expects
        self.resize = T.Resize((224, 224), antialias=True)

        action_min = PUSHT_ACTION_LOW.clone()
        action_max = PUSHT_ACTION_HIGH.clone()
        
        self.stats = {
            'action_min': action_min,
            'action_max': action_max,
            'action_space': 'swm_relative',
            'action_scale': PUSHT_ACTION_SCALE,
            'action_chunk_size': action_chunk_size,
        }
        
        # The diffusion-policy data stores absolute pixel targets. SWM PushT expects
        # relative controls: env target = agent_xy + action * action_scale.
        self.actions = absolute_to_relative_action(raw_actions, raw_states[:, :2])

        # Precompute episode boundaries for safe frame stacking
        self.ep_starts = np.zeros_like(self.episode_ends)
        self.ep_starts[1:] = self.episode_ends[:-1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Find which episode this index belongs to
        ep_idx = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = self.ep_starts[ep_idx]
        
        # Build the frame history without crossing episode boundaries
        frame_indices = []
        for i in range(self.frame_stack - 1, -1, -1):
            frame_idx = max(ep_start, idx - i)
            frame_indices.append(frame_idx)
            
        # Extract and stack images: Shape (FrameStack, C, H, W)
        obs_seq = self.images[frame_indices]
        obs_seq = self.resize(obs_seq)
        
        # Future action chunk, padded at episode boundaries without crossing into
        # the next demonstration.
        action_indices = []
        ep_end = self.episode_ends[ep_idx]
        for offset in range(self.action_chunk_size):
            action_idx = min(ep_end - 1, idx + offset)
            action_indices.append(action_idx)
        action_chunk = self.actions[action_indices]
        
        return obs_seq, action_chunk
