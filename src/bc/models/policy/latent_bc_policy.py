import torch.nn as nn

class LatentBCPolicy(nn.Module):
    def __init__(
        self,
        latent_dim=384,
        frame_stack=5,
        action_dim=2,
        hidden_dim=256,
        action_chunk_size=1,
    ):
        super().__init__()
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        
        # Total input size is the dimension of one latent vector * number of stacked frames
        input_dim = latent_dim * frame_stack
        output_dim = action_dim * action_chunk_size
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # Output is SWM-relative action chunks, trained against targets in [-1, 1].
        )

    def forward(self, stacked_latents):
        # Flatten the stacked latents from (Batch, Frames, LatentDim) to (Batch, Frames * LatentDim)
        batch_size = stacked_latents.shape[0]
        flattened = stacked_latents.reshape(batch_size, -1)
        flat_actions = self.net(flattened)
        return flat_actions.reshape(batch_size, self.action_chunk_size, self.action_dim)
