# PPO on `swm/PushT-v1` (from scratch)

A self-contained, CleanRL-style PPO implementation for the
[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel) Push-T
environment, where a circular agent pushes a T-block to a goal pose.

## Reward

Uses the environment's **native dense reward** — the negative L2 distance
between the current and goal full 7-D state:

```
reward = -‖goal_state − state‖₂
```

No reward shaping. This is a plain PPO baseline. Note that the full state
includes the agent's own (randomly sampled) goal position, so the task is to
drive the entire state — agent pose, block pose, and orientation — to the goal
snapshot. This is hard for pure from-scratch PPO (PushT is a contact-rich
manipulation task, which is why it is normally tackled with imitation learning);
treat these numbers as a baseline.

## Environment contract (what the code relies on)

| Aspect        | Detail                                                                 |
|---------------|-----------------------------------------------------------------------|
| Obs (dict)    | `state` (7,) = `[agent_xy, block_xy, block_angle, agent_vxy]`; `proprio` (4,) |
| **Goal**      | Re-randomized every episode, returned in `info["goal_state"]` (7,)     |
| Action        | `Box(-1, 1, (2,))`, relative velocity (PD-controlled, scaled ×100)     |
| Reward        | dense `-‖goal − state‖₂`                                               |
| Termination   | `terminated=True` **only on success** (pos err <20px, angle err <π/9) |
| Time limit    | **not built in** — added via `gym.make(..., max_episode_steps=200)`    |

Because the goal changes per episode, the policy is **goal-conditioned**: the
observation fed to the network is `concat(state, goal_state)` → 14-D
(`envs.GoalConditionedFlatten`).

## Layout

| File           | Purpose                                                            |
|----------------|-------------------------------------------------------------------|
| `config.py`    | `Config` dataclass — all hyperparameters                          |
| `envs.py`      | env factory + goal-conditioning / flattening wrapper              |
| `networks.py`  | `ActorCritic` MLP (diagonal Gaussian policy)                      |
| `utils.py`     | seeding, device, running mean/std obs & reward normalizers        |
| `ppo.py`       | `PPOTrainer` — rollout, GAE, clipped PPO update, checkpointing    |
| `train.py`     | CLI entry point (`python -m src.ppo.train`)                       |
| `evaluate.py`  | load a checkpoint, evaluate, optionally record MP4 videos         |

### Implementation notes

- Parallel envs are managed as a plain Python list (not a Gymnasium vector env)
  so that **time-limit truncation bootstrapping is explicit**: on truncation we
  add `gamma · V(terminal_obs)` to the reward; on success (termination) we do not.
- Observations are normalized with a shared running mean/std; rewards are scaled
  by the running std of the discounted return (gym-style).

## Usage

Run from the **project root** with the env that has `stable-worldmodel` installed
(here, the `dl_lab` conda env):

```bash
# quick smoke test
python -m src.ppo.train --total-timesteps 8192 --num-envs 4 --num-steps 128 \
    --num-minibatches 4 --update-epochs 4 --max-episode-steps 50

# full training run
python -m src.ppo.train --total-timesteps 3000000 --num-envs 8 --exp-name pusht_run1

# with Weights & Biases logging
python -m src.ppo.train --track --wandb-project offline-rl-with-le-wm

# evaluate + record videos
python -m src.ppo.evaluate --checkpoint runs/pusht_run1__seed1/final.pt \
    --episodes 20 --video
```

Checkpoints and videos are written under `runs/<exp_name>__seed<seed>/`.

Every `Config` field is a CLI flag (`snake_case` → `--kebab-case`); booleans get
a `--flag` / `--no-flag` pair.

## Logged metrics

`success_rate`, `final_distance` (mean terminal `‖goal − state‖`),
`episodic_return`, `episodic_length`, plus PPO diagnostics (`approx_kl`,
`clipfrac`, `explained_variance`).
