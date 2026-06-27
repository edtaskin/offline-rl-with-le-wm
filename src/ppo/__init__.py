"""From-scratch PPO for the stable-worldmodel ``swm/PushT-v1`` environment.

The environment exposes a dense reward (negative L2 distance to a per-episode
goal state), so a goal-conditioned MLP policy trained with vanilla PPO is a
sensible baseline. See ``config.py`` for hyperparameters and ``train.py`` for
the entry point (``python -m src.ppo.train``).
"""

from src.ppo.config import Config

__all__ = ["Config"]
