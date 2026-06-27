"""Hyperparameters for PPO on ``swm/PushT-v1``.

A single dataclass keeps every knob in one place; ``train.py`` exposes each
field as a command-line flag via ``tyro``-style parsing (here: argparse).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ----- experiment -----
    exp_name: str = "ppo_pusht"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    track: bool = False  # log to Weights & Biases
    wandb_project: str = "offline-rl-with-le-wm"
    wandb_entity: str | None = None
    capture_video: bool = False  # record eval videos during training
    save_dir: str = "runs"

    # ----- environment -----
    env_id: str = "swm/PushT-v1"
    max_episode_steps: int = 200

    # ----- training budget -----
    total_timesteps: int = 3_000_000
    num_envs: int = 8
    num_steps: int = 256  # rollout horizon per env -> batch = num_envs * num_steps

    # ----- optimization -----
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    # ----- network -----
    hidden_dims: tuple[int, ...] = (256, 256)

    # ----- normalization -----
    norm_obs: bool = True
    norm_reward: bool = True
    obs_clip: float = 10.0
    reward_clip: float = 10.0

    # ----- logging / checkpointing (in iterations) -----
    log_interval: int = 1
    eval_interval: int = 50
    eval_episodes: int = 10
    save_interval: int = 50

    # ----- derived (filled in __post_init__) -----
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = int(self.total_timesteps // self.batch_size)
