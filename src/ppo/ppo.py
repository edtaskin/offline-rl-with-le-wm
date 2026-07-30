"""Chunk-level PPO fine-tuning of a Latent BC agent on ``swm/PushT-v1``.

A from-scratch, CleanRL-style trainer where the observation is an image and the
"action" is an open-loop chunk of ``action_chunk_size`` env steps:

* One rollout entry = one chunk decision. The env advances ``k`` steps executing
  the chunk; the transition reward is the within-chunk discounted sum
  ``sum_j gamma**j r_j``; the inter-transition discount used by GAE is
  ``chunk_gamma = gamma**k``.
* The frozen encoder turns each frame into a latent once, during rollout; the
  dilated stack of latents (see :class:`src.ppo.env.LatentHistory`) is the
  policy/value input and is what we store. The PPO update runs on stored latents,
  so the ViT never appears in the optimization loop.
* Parallel envs are managed as a plain Python list (not a Gymnasium vector env)
  so that time-limit truncation (not success) unambiguously bootstraps
  ``gamma**(j+1) * V(terminal)`` before the env is reset.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Attach a timestamped stream handler to the module logger (idempotent)."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # own handler; don't double-print via the root logger

from src.ppo.agent import build_latent_agent
from src.ppo.config import LatentConfig
from src.ppo.env import LatentHistory, make_latent_env, success_from_info
from src.representations.lewm import LeWMEncoder
from src.ppo.utils import RewardNormalizer, get_device, set_seed


class LatentPPOTrainer:
    def __init__(self, cfg: LatentConfig, encoder=None):
        self.cfg = cfg
        _configure_logging()
        set_seed(cfg.seed, cfg.torch_deterministic)
        self.device = get_device(cfg.device)

        # Parallel image envs, managed manually (see module docstring).
        self.envs = [
            make_latent_env(
                env_id=cfg.env_id,
                seed=cfg.seed,
                idx=i,
                max_episode_steps=cfg.max_episode_steps,
                observation_resolution=cfg.observation_resolution,
                fixed_target=cfg.fixed_target,
                fixed_target_block_success=cfg.fixed_target_block_success,
                agent_block_coef=cfg.agent_block_coef,
                block_start_near_goal=cfg.block_start_near_goal,
                block_start_radius=cfg.block_start_radius,
                reward_mode=cfg.reward_mode,
            )()
            for i in range(cfg.num_envs)
        ]

        # Frozen encoder (allow injection for testing) + BC-initialized agent.
        if encoder is None:
            encoder = LeWMEncoder.from_checkpoint(
                self.device,
                cfg.encoder_checkpoint,
                latent_dim=cfg.latent_dim,
            )
        self.encoder = encoder
        self.agent = build_latent_agent(
            encoder=encoder,
            latent_dim=cfg.latent_dim,
            frame_stack=cfg.frame_stack,
            action_dim=cfg.action_dim,
            action_chunk_size=cfg.action_chunk_size,
            hidden_dim=cfg.hidden_dim,
            init_log_std=cfg.init_log_std,
            bc_checkpoint_path=cfg.bc_checkpoint,
            device=self.device,
        )
        # Load frozen BC policy if KL penalty required
        self.bc_ref_policy = None
        if cfg.BC_KL_penalty:
            if cfg.bc_checkpoint is None:
                raise ValueError("BC_KL_penalty requires cfg.bc_checkpoint")
            self.bc_ref_policy = LatentBCPolicy(
                latent_dim=cfg.latent_dim,
                frame_stack=cfg.frame_stack,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                action_chunk_size=cfg.action_chunk_size,
            ).to(self.device)
            self.bc_ref_policy.load_state_dict(
                torch.load(cfg.bc_checkpoint, map_location=self.device)
            )
            self.bc_ref_policy.eval()
            for param in self.bc_ref_policy.parameters():
                param.requires_grad = False
        if cfg.anneal_log_std:
            # Schedule exploration instead of learning it: freeze the parameter
            # (so it is excluded from ``trainable`` below) and set it per
            # iteration in ``train`` via ``_set_log_std``.
            self.agent.actor.log_std.requires_grad_(False)
            self._set_log_std(cfg.init_log_std)

        trainable = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=cfg.learning_rate, eps=1e-5)

        self.reward_norm = (
            RewardNormalizer(cfg.num_envs, cfg.chunk_gamma, clip=cfg.reward_clip)
            if cfg.norm_reward
            else None
        )

        self.histories = [
            LatentHistory(cfg.frame_stack, cfg.frame_stride) for _ in range(cfg.num_envs)
        ]

        self.global_step = 0  # counts real env steps
        self.start_time = time.time()
        self._ep_returns: deque[float] = deque(maxlen=100)
        self._ep_lengths: deque[float] = deque(maxlen=100)
        self._ep_success: deque[float] = deque(maxlen=100)
        self._ep_final_dist: deque[float] = deque(maxlen=100)

        # Best / second-best checkpoints ranked by success rate (held-out when
        # cfg.eval_interval > 0, else the rolling window).
        self._best_success = -float("inf")
        self._second_best_success = -float("inf")
        self._eval_env = None  # dedicated held-out env, built lazily on first eval

        run_stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
        self.run_dir = Path(cfg.save_dir) / f"{cfg.exp_name}__seed{cfg.seed}" / run_stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None  # optional wandb run, set in train.py

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _set_log_std(self, value: float) -> None:
        """Set the (frozen) actor log_std to a constant; used by log_std annealing."""
        self.agent.actor.log_std.data.fill_(float(value))

    @torch.no_grad()
    def _encode_obs(self, obs_batch: np.ndarray) -> torch.Tensor:
        """``[n, H, W, C]`` (or ``[H, W, C]``) uint8 frames -> ``[n, latent_dim]``."""
        t = torch.as_tensor(np.asarray(obs_batch), device=self.device)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.permute(0, 3, 1, 2).contiguous()
        return self.encoder(t)

    def _stacked_latents(self) -> torch.Tensor:
        """Current dilated latent stack for every env: ``[num_envs, F, latent_dim]``."""
        return torch.stack([h.stacked() for h in self.histories], dim=0)

    def _record_episode(self, info: dict, terminated: bool, last_reward: float) -> None:
        if "episode" in info:
            self._ep_returns.append(float(info["episode"]["r"]))
            self._ep_lengths.append(float(info["episode"]["l"]))
        self._ep_success.append(success_from_info(info, terminated))
        # Native reward is -distance, so the final distance is -reward.
        self._ep_final_dist.append(float(-last_reward))

    # -------------------------------------------------------------- rollouts
    def _reset_all(self) -> None:
        for i, (env, hist) in enumerate(zip(self.envs, self.histories)):
            obs, _ = env.reset(seed=self.cfg.seed + i)
            hist.clear()
            hist.append(self._encode_obs(obs)[0])

    def collect_rollout(self, done: np.ndarray):
        cfg = self.cfg
        n, e, k = cfg.num_chunks, cfg.num_envs, cfg.action_chunk_size
        D, F = cfg.latent_dim, cfg.frame_stack

        b_latents = torch.zeros((n, e, F, D), dtype=torch.float32, device=self.device)
        b_actions = torch.zeros(
            (n, e, k, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        b_logprobs = np.zeros((n, e), dtype=np.float32)
        b_rewards = np.zeros((n, e), dtype=np.float32)
        b_dones = np.zeros((n, e), dtype=np.float32)
        b_values = np.zeros((n, e), dtype=np.float32)

        for step in range(n):
            stacked = self._stacked_latents()  # [e, F, D]
            b_latents[step] = stacked
            b_dones[step] = done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value_from_latents(
                    stacked
                )
            b_actions[step] = action
            b_logprobs[step] = logprob.cpu().numpy()
            b_values[step] = value.cpu().numpy()

            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()  # [e, k, adim]

            chunk_reward = np.zeros(e, dtype=np.float64)
            step_done = np.zeros(e, dtype=np.float32)
            bootstrap = np.zeros(e, dtype=np.float32)
            active = np.ones(e, dtype=bool)

            for j in range(k):
                active_idx = np.nonzero(active)[0]
                if active_idx.size == 0:
                    break

                next_obs = {}
                for i in active_idx:
                    no, r, term, trunc, info = self.envs[i].step(action_np[i, j])
                    self.global_step += 1
                    chunk_reward[i] += (cfg.gamma**j) * float(r)
                    next_obs[i] = (no, float(r), bool(term), bool(trunc), info)

                # Batch-encode the active envs' next frames.
                latents = self._encode_obs(np.stack([next_obs[i][0] for i in active_idx]))

                for pos, i in enumerate(active_idx):
                    no, r, term, trunc, info = next_obs[i]
                    self.histories[i].append(latents[pos])
                    if not (term or trunc):
                        continue

                    active[i] = False
                    step_done[i] = 1.0
                    if trunc and not term:
                        # Bootstrap the value of the genuine terminal observation
                        # (its dilated stack now ends at this frame) before reset.
                        with torch.no_grad():
                            v_term = self.agent.get_value_from_latents(
                                self.histories[i].stacked().unsqueeze(0)
                            ).item()
                        bootstrap[i] = (cfg.gamma ** (j + 1)) * v_term

                    self._record_episode(info, term, r)
                    reset_obs, _ = self.envs[i].reset()
                    self.histories[i].clear()
                    self.histories[i].append(self._encode_obs(reset_obs)[0])

            if self.reward_norm is not None:
                rewards = self.reward_norm.normalize(chunk_reward, step_done)
            else:
                rewards = chunk_reward.astype(np.float32)
            b_rewards[step] = rewards + bootstrap
            done = step_done

        return b_latents, b_actions, b_logprobs, b_rewards, b_dones, b_values, done

    def compute_gae(self, rewards, values, dones, next_done):
        cfg = self.cfg
        n = cfg.num_chunks
        with torch.no_grad():
            next_value = self.agent.get_value_from_latents(self._stacked_latents())
            next_value = next_value.cpu().numpy()

        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(n)):
            if t == n - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + cfg.chunk_gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = (
                delta + cfg.chunk_gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            )
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    # ---------------------------------------------------------------- update
    def update(self, b_latents, b_actions, b_logprobs, b_values, advantages, returns):
        cfg = self.cfg
        F, D, k = cfg.frame_stack, cfg.latent_dim, cfg.action_chunk_size

        latents = b_latents.reshape(-1, F, D)
        actions = b_actions.reshape(-1, k, cfg.action_dim)
        logprobs = torch.as_tensor(b_logprobs.reshape(-1), device=self.device)
        values = torch.as_tensor(b_values.reshape(-1), device=self.device)
        advantages_t = torch.as_tensor(advantages.reshape(-1), device=self.device)
        returns_t = torch.as_tensor(returns.reshape(-1), device=self.device)

        inds = np.arange(cfg.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0, device=self.device)
        bc_kl_loss = torch.tensor(0.0, device=self.device)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = inds[start : start + cfg.minibatch_size]

                _, newlogprob, entropy, newvalue = (
                    self.agent.get_action_and_value_from_latents(latents[mb], actions[mb])
                )
                logratio = newlogprob - logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_adv = advantages_t[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_unclipped = (newvalue - returns_t[mb]) ** 2
                    v_clipped = values[mb] + torch.clamp(
                        newvalue - values[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped = (v_clipped - returns_t[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns_t[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                if cfg.BC_KL_penalty:
                    if self.bc_ref_policy is None:
                        raise RuntimeError("BC_KL_penalty is enabled without a BC reference policy")
                    current_action_mean = self.agent.actor.bc_policy(latents[mb])
                    with torch.no_grad():
                        bc_action_mean = self.bc_ref_policy(latents[mb])
                    bc_kl_loss = ((current_action_mean - bc_action_mean) ** 2).mean()
                else:
                    bc_kl_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    pg_loss
                    - cfg.ent_coef * entropy_loss
                    + cfg.vf_coef * v_loss
                    + cfg.BC_KL_penalty_coef * bc_kl_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.reshape(-1), returns.reshape(-1)
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "loss/policy": pg_loss.item(),
            "loss/value": v_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/bc_kl": bc_kl_loss.item(),
            "loss/approx_kl": approx_kl.item(),
            "loss/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "loss/explained_variance": float(explained_var),
        }

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, name: str = "latest", success_rate: float | None = None) -> Path:
        path = self.run_dir / f"{name}.pt"
        # Drop the (frozen, large) encoder weights; it is reloaded separately.
        agent_state = {
            key: value
            for key, value in self.agent.state_dict().items()
            if not key.startswith(("actor.encoder.", "critic.encoder."))
        }
        ckpt = {
            "agent": agent_state,
            "config": self.cfg.__dict__,
            "global_step": self.global_step,
            "success_rate": success_rate,
            "contract": {
                "frame_stack": self.cfg.frame_stack,
                "frame_stride": self.cfg.frame_stride,
                "action_chunk_size": self.cfg.action_chunk_size,
                "latent_dim": self.cfg.latent_dim,
                "hidden_dim": self.cfg.hidden_dim,
                "action_dim": self.cfg.action_dim,
            },
        }
        if self.reward_norm is not None:
            ckpt["reward_norm"] = self.reward_norm.state_dict()
        torch.save(ckpt, path)
        return path

    def _current_success_rate(self) -> float:
        return float(np.mean(self._ep_success)) if self._ep_success else float("nan")

    @torch.no_grad()
    def _evaluate_heldout(self) -> dict:
        """Evaluate the in-memory agent through the canonical PushT runner."""
        from src.evaluation.agents import PPOComponents, make_ppo_evaluation_agent
        from src.evaluation.pusht import PushTEvalConfig, run_evaluation

        cfg = self.cfg
        if self._eval_env is None:
            self._eval_env = make_latent_env(
                env_id=cfg.env_id,
                seed=cfg.eval_seed,
                idx=0,
                max_episode_steps=cfg.max_episode_steps,
                observation_resolution=cfg.observation_resolution,
                record_stats=True,
                fixed_target=True,
                fixed_target_block_success=cfg.fixed_target_block_success,
                block_start_near_goal=cfg.block_start_near_goal,
                block_start_radius=cfg.block_start_radius,
            )()
        contract = {
            "frame_stack": cfg.frame_stack,
            "frame_stride": cfg.frame_stride,
            "action_chunk_size": cfg.action_chunk_size,
            "latent_dim": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "action_dim": cfg.action_dim,
        }
        adapter = make_ppo_evaluation_agent(
            components=PPOComponents(
                encoder=self.encoder,
                agent=self.agent,
                contract=contract,
                config=cfg.__dict__,
            ),
            deterministic=True,
            execution_mode="open-loop",
        )
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            result = run_evaluation(
                adapter,
                PushTEvalConfig(
                    env_id=cfg.env_id,
                    episodes=cfg.eval_episodes,
                    seed=cfg.eval_seed,
                    max_episode_steps=cfg.max_episode_steps,
                    observation_resolution=cfg.observation_resolution,
                    fixed_target_block_success=cfg.fixed_target_block_success,
                    block_start_radius=(
                        cfg.block_start_radius if cfg.block_start_near_goal else None
                    ),
                ),
                env=self._eval_env,
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        logger.info(
            "Held-out eval | success %4.2f | len %5.1f | (%d eps, seed %d)",
            result.summary["success_rate"],
            result.summary["mean_length"],
            cfg.eval_episodes,
            cfg.eval_seed,
        )
        return {
            "success_rate": result.summary["success_rate"],
            "mean_length": result.summary["mean_length"],
            "episodes": cfg.eval_episodes,
        }

    def _log_heldout(self, iteration: int, stats: dict) -> None:
        logger.info(
            "iter %d/%d | HELD-OUT | success %4.2f | len %5.1f | (%d eps, seed %d)",
            iteration,
            self.cfg.num_iterations,
            stats["success_rate"],
            stats["mean_length"],
            stats["episodes"],
            self.cfg.eval_seed,
        )
        if self.writer is not None:
            self.writer.log(
                {
                    "eval/heldout_success": stats["success_rate"],
                    "eval/heldout_length": stats["mean_length"],
                },
                step=self.global_step,
            )

    def _update_best_checkpoints(self, success_rate: float | None = None) -> None:
        """Save ``best.pt`` / ``second_best.pt`` ranked by success rate.

        ``success_rate`` defaults to the rolling-window estimate; the training
        loop passes the held-out eval success instead when ``eval_interval > 0``.
        When a new best is reached, the previous ``best.pt`` is demoted to
        ``second_best.pt`` (copying the file preserves those exact weights, which
        are no longer in memory once training has moved on).
        """
        if success_rate is None:
            success_rate = self._current_success_rate()
        if not np.isfinite(success_rate):
            return  # no completed episodes yet

        if success_rate > self._best_success:
            best_path = self.run_dir / "best.pt"
            if best_path.exists():
                shutil.copyfile(best_path, self.run_dir / "second_best.pt")
                logger.info(
                    "New best success %.4f > %.4f; demoted previous best to second_best.pt",
                    success_rate,
                    self._best_success,
                )
                self._second_best_success = self._best_success
            self.save_checkpoint("best", success_rate=success_rate)
            self._best_success = success_rate
        elif success_rate > self._second_best_success:
            logger.info(
                "New second-best success %.4f > %.4f",
                success_rate,
                self._second_best_success,
            )
            self.save_checkpoint("second_best", success_rate=success_rate)
            self._second_best_success = success_rate

    # ----------------------------------------------------------------- train
    def train(self) -> None:
        cfg = self.cfg
        self._reset_all()
        done = np.zeros(cfg.num_envs, dtype=np.float32)

        for iteration in range(1, cfg.num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            if cfg.anneal_log_std:
                # frac: 1 at the first iteration -> ~0 at the last, so log_std
                # goes init_log_std -> final_log_std.
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                self._set_log_std(
                    cfg.final_log_std + frac * (cfg.init_log_std - cfg.final_log_std)
                )

            (
                b_latents,
                b_actions,
                b_logprobs,
                b_rewards,
                b_dones,
                b_values,
                done,
            ) = self.collect_rollout(done)

            advantages, returns = self.compute_gae(b_rewards, b_values, b_dones, done)
            stats = self.update(
                b_latents, b_actions, b_logprobs, b_values, advantages, returns
            )

            if iteration % cfg.log_interval == 0:
                self._log(iteration, stats)

            if cfg.eval_interval > 0:
                # Honest selection: rank checkpoints by held-out success, refreshed
                # every eval_interval iterations.
                if iteration % cfg.eval_interval == 0:
                    eval_stats = self._evaluate_heldout()
                    self._log_heldout(iteration, eval_stats)
                    self._update_best_checkpoints(eval_stats["success_rate"])
            else:
                # Track best / second-best every iteration so peaks are never missed.
                self._update_best_checkpoints()

            if iteration % cfg.save_interval == 0:
                self.save_checkpoint("latest", success_rate=self._current_success_rate())

        final_success = None
        if cfg.eval_interval > 0:
            eval_stats = self._evaluate_heldout()
            self._log_heldout(cfg.num_iterations, eval_stats)
            self._update_best_checkpoints(eval_stats["success_rate"])
            final_success = eval_stats["success_rate"]
        self.save_checkpoint(
            "final",
            success_rate=final_success if final_success is not None else self._current_success_rate(),
        )
        for env in self.envs:
            env.close()
        if self._eval_env is not None:
            self._eval_env.close()

    def _log(self, iteration: int, stats: dict) -> None:
        elapsed = max(time.time() - self.start_time, 1e-6)
        sps = int(self.global_step / elapsed)
        ep_ret = np.mean(self._ep_returns) if self._ep_returns else float("nan")
        ep_len = np.mean(self._ep_lengths) if self._ep_lengths else float("nan")
        success = self._current_success_rate()
        final_dist = np.mean(self._ep_final_dist) if self._ep_final_dist else float("nan")

        logger.info(
            "iter %d/%d | step %d | sps %d | ret %8.1f | len %5.1f | "
            "success %4.2f | dist %6.1f | kl %.4f | ev %.2f",
            iteration,
            self.cfg.num_iterations,
            self.global_step,
            sps,
            ep_ret,
            ep_len,
            success,
            final_dist,
            stats["loss/approx_kl"],
            stats["loss/explained_variance"],
        )

        if self.writer is not None:
            metrics = {
                "charts/episodic_return": ep_ret,
                "charts/episodic_length": ep_len,
                "charts/success_rate": success,
                "charts/final_distance": final_dist,
                "charts/SPS": sps,
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                **stats,
            }
            self.writer.log(metrics, step=self.global_step)
