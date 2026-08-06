"""Hyperparameters for chunk-level latent PPO fine-tuning on ``swm/PushT-v1``.

One PPO transition corresponds to one open-loop *action chunk* of
``action_chunk_size`` env steps. ``gamma`` is the per-env-step discount; the
effective per-transition discount is ``chunk_gamma = gamma ** action_chunk_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.envs import PUSHT_RENDER_SHAPE


@dataclass
class LatentConfig:
    # ----- experiment -----
    exp_name: str = "latent_ppo_pusht_sparse_circle"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    track: bool = False
    wandb_project: str = "ppo-training"
    wandb_entity: str | None = "imendezval-university-freiburg"
    save_dir: str = "runs"

    # ----- environment -----
    env_id: str = "swm/PushT-v1"
    max_episode_steps: int = 300
    observation_resolution: int = PUSHT_RENDER_SHAPE[0]
    fixed_target: bool = False
    fixed_target_block_success: bool = True
    # Reward shaping (fixed_target only): subtract agent_block_coef * ||agent-block||
    # from the reward so the policy stays engaged with the block. 0.0 = off.
    agent_block_coef: float = 0.0
    # Start each episode with the block within block_start_radius pixels of the
    # green T (goal) center (see PushTBlockStartNearGoalWrapper). Shortens the
    # block-transport distance into a controllable range.
    block_start_near_goal: bool = True
    block_start_radius: float = 200.0
    # Reward type: "dense" = native env reward; "sparse" = 1.0 on success, else 0.0.
    reward_mode: str = "sparse"

    # ----- BC prior / encoder -----
    bc_checkpoint: str = (
        "hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth"
    )
    bc_stats: str = (
        "hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth"
    )
    encoder_checkpoint: str | None = None  # None -> default swm cache path

    # ----- agent contract (defaults match the published BC checkpoint; the entry
    # point overrides these from ``bc_stats`` when present) -----
    frame_stack: int = 3
    frame_stride: int = 5
    action_chunk_size: int = 5
    latent_dim: int = 192
    hidden_dim: int = 256
    action_dim: int = 2
    init_log_std: float = -2.0  # exploration std ~= 0.6 in [-1, 1] action space
    # Anneal exploration instead of learning it: freeze log_std and interpolate
    # init_log_std -> final_log_std over training so PPO sharpens the *deployed*
    # (mean) policy under shrinking noise. final_log_std=-3.5 => sigma ~= 0.03.
    anneal_log_std: bool = False
    final_log_std: float = -3.5

    # ----- training budget (total_timesteps counts real env steps) -----
    total_timesteps: int = 1_000_000
    num_envs: int = 8
    num_chunks: int = 64  # rollout horizon in chunks per env

    # ----- optimization -----
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99  # per-env-step discount
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    bc_penalty: bool = False
    bc_penalty_coef: float = 0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    # ----- reward normalization -----
    norm_reward: bool = True
    reward_clip: float = 10.0

    # ----- logging / checkpointing (in iterations) -----
    log_interval: int = 1
    save_interval: int = 25
    # Keep a permanent, step-tagged copy of the agent every ``snapshot_interval``
    # iterations (``snapshot_step<env_steps>_it<iteration>.pt``). Unlike
    # ``latest.pt`` these are never overwritten, which is what makes an
    # interaction-budget curve (success vs env steps consumed) possible after the
    # fact. 0 = off.
    snapshot_interval: int = 0

    # ----- held-out evaluation (deterministic, fixed seeds) -----
    # Honest best-checkpoint selection: every ``eval_interval`` iterations, run
    # ``eval_episodes`` deterministic (mean-action, open-loop) episodes on seeds
    # ``[eval_seed, eval_seed + N)`` and select ``best.pt`` by that success --
    # matching the canonical evaluator in ``src/evaluation/evaluate_pusht.py``.
    # ``0`` disables it and falls back to the rolling-window (deque) selection.
    eval_interval: int = 0
    eval_episodes: int = 20
    eval_seed: int = 0

    # ----- optional Hugging Face upload of the final checkpoint -----
    push_to_hf: bool = False
    hf_repo_id: str | None = None
    hf_repo_type: str = "model"
    hf_private: bool = False
    hf_token: str | None = None
    hf_revision: str | None = None
    hf_path_prefix: str | None = None
    hf_commit_message: str = "Upload latent PPO checkpoint"

    # ----- derived (filled in __post_init__) -----
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)
    chunk_gamma: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.observation_resolution < 1:
            raise ValueError("observation_resolution must be positive")
        self.batch_size = int(self.num_envs * self.num_chunks)
        self.minibatch_size = max(1, int(self.batch_size // self.num_minibatches))
        steps_per_iter = self.batch_size * self.action_chunk_size
        self.num_iterations = max(1, int(self.total_timesteps // steps_per_iter))
        self.chunk_gamma = float(self.gamma**self.action_chunk_size)
