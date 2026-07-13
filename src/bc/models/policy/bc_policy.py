import torch.nn as nn

class BCMLPPolicy(nn.Module):
    """
    Simple MLP policy for continuous action regression.
    Outputs mean actions directly for standard Behavior Cloning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """
        Maps normalized states directly to normalized actions.
        """
        return self.net(state)