import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


PUSHT_ACTION_LOW = torch.tensor([0.0, 0.0], dtype=torch.float32)
PUSHT_ACTION_HIGH = torch.tensor([512.0, 512.0], dtype=torch.float32)


def normalize_action(action, action_min=PUSHT_ACTION_LOW, action_max=PUSHT_ACTION_HIGH):
    action_range = torch.clamp(action_max - action_min, min=1e-6)
    return 2.0 * (action - action_min) / action_range - 1.0


def unnormalize_action(norm_action, action_min=PUSHT_ACTION_LOW, action_max=PUSHT_ACTION_HIGH):
    return 0.5 * (norm_action + 1.0) * (action_max - action_min) + action_min


class PushTLeWMDataset(Dataset):
    def __init__(self, data_path, frame_stack=5):
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")

        data = np.load(data_path, allow_pickle=True)
        
        # Load images and ensure (N, C, H, W) format
        raw_images = data['images']
        if raw_images.shape[-1] == 3:
            raw_images = np.transpose(raw_images, (0, 3, 1, 2))
            
        self.images = torch.tensor(raw_images, dtype=torch.float32) / 255.0
        raw_actions = torch.tensor(data['actions'], dtype=torch.float32)
        
        self.episode_ends = data['episode_ends']
        self.frame_stack = frame_stack

        if len(self.images) != len(raw_actions):
            raise ValueError(f"images/actions length mismatch: {len(self.images)} vs {len(raw_actions)}")
        if raw_actions.shape[-1] != 2:
            raise ValueError(f"expected 2D PushT actions, got shape {tuple(raw_actions.shape)}")
        if len(self.episode_ends) == 0 or int(self.episode_ends[-1]) != len(self.images):
            raise ValueError("episode_ends must be non-empty and end at the dataset length")
        
        # Resize to match what LeWM expects
        self.resize = T.Resize((224, 224), antialias=True)

        # The PushT physics arena is strictly 512x512
        action_min = PUSHT_ACTION_LOW.clone()
        action_max = PUSHT_ACTION_HIGH.clone()
        
        self.stats = {'action_min': action_min, 'action_max': action_max}
        
        # Normalize expert actions to strictly map to [-1.0, 1.0] relative to the 512x512 box
        self.actions = normalize_action(raw_actions, action_min, action_max)

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
        
        # Single target action
        action = self.actions[idx]
        
        return obs_seq, action
