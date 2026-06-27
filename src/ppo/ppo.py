"""PPO trainer for ``swm/PushT-v1``.

A from-scratch, CleanRL-style implementation. We keep a plain list of parallel
environments and manage resets manually (rather than relying on Gymnasium's
vector autoreset semantics) so that bootstrapping at time-limit truncation is
unambiguous:

* ``terminated`` (success) -> no bootstrap, value target is the reward only.
* ``truncated`` (time limit) -> bootstrap ``gamma * V(terminal_obs)`` into the
  reward, since the episode was cut artificially.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ppo.config import Config
from src.ppo.envs import make_env
from src.ppo.networks import ActorCritic
from src.ppo.utils import (
    ObsNormalizer,
    RewardNormalizer,
    get_device,
    set_seed,
)


class PPOTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed, cfg.torch_deterministic)
        self.device = get_device(cfg.device)

        # Build parallel environments (managed manually, see module docstring).
        self.envs = [
            make_env(
                cfg.env_id,
                seed=cfg.seed,
                idx=i,
                max_episode_steps=cfg.max_episode_steps,
            )()
            for i in range(cfg.num_envs)
        ]
        self.obs_dim = int(self.envs[0].observation_space.shape[0])
        self.act_dim = int(self.envs[0].action_space.shape[0])

        self.agent = ActorCritic(self.obs_dim, self.act_dim, cfg.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        self.obs_norm = (
            ObsNormalizer((self.obs_dim,), clip=cfg.obs_clip) if cfg.norm_obs else None
        )
        self.reward_norm = (
            RewardNormalizer(cfg.num_envs, cfg.gamma, clip=cfg.reward_clip)
            if cfg.norm_reward
            else None
        )

        self.global_step = 0
        self.start_time = time.time()
        self._ep_returns: deque[float] = deque(maxlen=100)
        self._ep_lengths: deque[float] = deque(maxlen=100)
        self._ep_success: deque[float] = deque(maxlen=100)
        self._ep_final_dist: deque[float] = deque(maxlen=100)

        self.run_dir = Path(cfg.save_dir) / f"{cfg.exp_name}__seed{cfg.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None  # optional wandb run, set in train.py

    # ------------------------------------------------------------------ utils
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _norm_obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if self.obs_norm is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norm.normalize(obs, update=update)

    # -------------------------------------------------------------- rollouts
    def _reset_all(self) -> np.ndarray:
        raw = np.stack(
            [env.reset(seed=self.cfg.seed + i)[0] for i, env in enumerate(self.envs)]
        )
        return self._norm_obs(raw, update=True)

    def collect_rollout(self, obs: np.ndarray):
        """Run ``num_steps`` of interaction; return buffers + next obs/done."""
        cfg = self.cfg
        n, e = cfg.num_steps, cfg.num_envs

        b_obs = np.zeros((n, e, self.obs_dim), dtype=np.float32)
        b_actions = np.zeros((n, e, self.act_dim), dtype=np.float32)
        b_logprobs = np.zeros((n, e), dtype=np.float32)
        b_rewards = np.zeros((n, e), dtype=np.float32)
        b_dones = np.zeros((n, e), dtype=np.float32)
        b_values = np.zeros((n, e), dtype=np.float32)

        done = np.zeros(e, dtype=np.float32)

        for step in range(n):
            self.global_step += e
            b_obs[step] = obs
            b_dones[step] = done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self._to_tensor(obs))
            action_np = action.cpu().numpy()
            b_actions[step] = action_np
            b_logprobs[step] = logprob.cpu().numpy()
            b_values[step] = value.cpu().numpy()

            raw_rewards = np.zeros(e, dtype=np.float32)
            step_done = np.zeros(e, dtype=np.float32)
            bootstrap = np.zeros(e, dtype=np.float32)
            next_raw = np.zeros((e, self.obs_dim), dtype=np.float32)

            for i, env in enumerate(self.envs):
                no, r, term, trunc, info = env.step(action_np[i])
                raw_rewards[i] = r
                d = bool(term or trunc)
                step_done[i] = float(d)

                # Time-limit truncation without success: bootstrap the value of
                # the genuine terminal observation before it is reset away.
                if trunc and not term:
                    term_obs = self._norm_obs(no[None], update=False)
                    with torch.no_grad():
                        v_term = self.agent.get_value(self._to_tensor(term_obs)).item()
                    bootstrap[i] = cfg.gamma * v_term

                if d:
                    if "episode" in info:
                        self._ep_returns.append(float(info["episode"]["r"]))
                        self._ep_lengths.append(float(info["episode"]["l"]))
                    self._ep_success.append(float(info.get("is_success", False)))
                    # Native reward is -distance, so the final distance is -reward.
                    self._ep_final_dist.append(float(-r))
                    no, _ = env.reset()

                next_raw[i] = no

            # Normalize rewards (shared across envs), then add value bootstrap.
            if self.reward_norm is not None:
                rewards = self.reward_norm.normalize(raw_rewards, step_done)
            else:
                rewards = raw_rewards
            rewards = rewards + bootstrap

            b_rewards[step] = rewards
            obs = self._norm_obs(next_raw, update=True)
            done = step_done

        return b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, obs, done

    def compute_gae(self, rewards, values, dones, next_obs, next_done):
        cfg = self.cfg
        n = cfg.num_steps
        with torch.no_grad():
            next_value = (
                self.agent.get_value(self._to_tensor(next_obs)).squeeze(-1).cpu().numpy()
            )

        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(n)):
            if t == n - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    # ---------------------------------------------------------------- update
    def update(self, b_obs, b_actions, b_logprobs, b_values, advantages, returns):
        cfg = self.cfg
        obs = self._to_tensor(b_obs.reshape(-1, self.obs_dim))
        actions = self._to_tensor(b_actions.reshape(-1, self.act_dim))
        logprobs = self._to_tensor(b_logprobs.reshape(-1))
        values = self._to_tensor(b_values.reshape(-1))
        advantages = self._to_tensor(advantages.reshape(-1))
        returns = self._to_tensor(returns.reshape(-1))

        inds = np.arange(cfg.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = inds[start : start + cfg.minibatch_size]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    obs[mb], actions[mb]
                )
                logratio = newlogprob - logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy (clipped surrogate) loss.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped).
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - returns[mb]) ** 2
                    v_clipped = values[mb] + torch.clamp(
                        newvalue - values[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped = (v_clipped - returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.reshape(-1), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "loss/policy": pg_loss.item(),
            "loss/value": v_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/approx_kl": approx_kl.item(),
            "loss/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "loss/explained_variance": float(explained_var),
        }

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, name: str = "latest") -> Path:
        path = self.run_dir / f"{name}.pt"
        ckpt = {
            "agent": self.agent.state_dict(),
            "config": self.cfg.__dict__,
            "global_step": self.global_step,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }
        if self.obs_norm is not None:
            ckpt["obs_norm"] = self.obs_norm.state_dict()
        if self.reward_norm is not None:
            ckpt["reward_norm"] = self.reward_norm.state_dict()
        torch.save(ckpt, path)
        return path

    # ----------------------------------------------------------------- train
    def train(self) -> None:
        cfg = self.cfg
        obs = self._reset_all()
        next_done = np.zeros(cfg.num_envs, dtype=np.float32)

        for iteration in range(1, cfg.num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            (
                b_obs,
                b_actions,
                b_logprobs,
                b_rewards,
                b_dones,
                b_values,
                obs,
                next_done,
            ) = self.collect_rollout(obs)

            advantages, returns = self.compute_gae(
                b_rewards, b_values, b_dones, obs, next_done
            )
            stats = self.update(
                b_obs, b_actions, b_logprobs, b_values, advantages, returns
            )

            if iteration % cfg.log_interval == 0:
                self._log(iteration, stats)
            if iteration % cfg.save_interval == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        for env in self.envs:
            env.close()

    def _log(self, iteration: int, stats: dict) -> None:
        sps = int(self.global_step / (time.time() - self.start_time))
        ep_ret = np.mean(self._ep_returns) if self._ep_returns else float("nan")
        ep_len = np.mean(self._ep_lengths) if self._ep_lengths else float("nan")
        success = np.mean(self._ep_success) if self._ep_success else float("nan")
        final_dist = np.mean(self._ep_final_dist) if self._ep_final_dist else float("nan")

        print(
            f"iter {iteration}/{self.cfg.num_iterations} | "
            f"step {self.global_step} | sps {sps} | "
            f"ret {ep_ret:8.1f} | len {ep_len:5.1f} | "
            f"success {success:4.2f} | dist {final_dist:6.1f} | "
            f"kl {stats['loss/approx_kl']:.4f} | ev {stats['loss/explained_variance']:.2f}"
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
