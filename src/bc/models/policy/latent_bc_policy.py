import torch.nn as nn

class LatentBCPolicy(nn.Module):
    def __init__(self, latent_dim=384, frame_stack=5, action_dim=2, hidden_dim=256):
        super().__init__()
        
        # Total input size is the dimension of one latent vector * number of stacked frames
        input_dim = latent_dim * frame_stack
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # Output is raw logits, normalized between -1 and 1 via loss function
        )

    def forward(self, stacked_latents):
        # Flatten the stacked latents from (Batch, Frames, LatentDim) to (Batch, Frames * LatentDim)
        batch_size = stacked_latents.shape[0]
        flattened = stacked_latents.reshape(batch_size, -1)
        return self.net(flattened)