# Latent PPO on `swm/PushT-v1`

Chunk-level PPO fine-tuning of a Latent BC policy on the
[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel) Push-T
environment. A frozen LeWM (JEPA ViT) encoder turns frames into latents; the
BC-initialized policy emits open-loop action chunks that PPO fine-tunes together
with an exploration `log_std` and a separate value head.

## Layout

| File              | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `config.py`       | `LatentConfig` dataclass — all hyperparameters                       |
| `env.py`          | image-observation env factory + `LatentHistory` (dilated latent stack) |
| `agent.py`        | `LatentPPOAgent` (BC-prior actor, value head) + `build_latent_agent` |
| `lewm_encoder.py` | frozen LeWM ViT encoder wrapper (`LeWMLatentEncoder`)                |
| `utils.py`        | seeding, device selection, reward normalizer                         |
| `ppo.py`          | `LatentPPOTrainer` — chunk rollout, GAE, clipped PPO update, checkpointing |
| `train.py`        | CLI entry point (`python -m src.ppo.train`)                          |
| `evaluate.py`     | load a checkpoint, evaluate (open-loop / receding-horizon / temporal ensemble), record MP4s |

## How it works

- One PPO transition = one open-loop *action chunk* of `action_chunk_size` env
  steps. The transition reward is the within-chunk discounted sum
  `sum_j gamma**j r_j`; GAE uses `chunk_gamma = gamma**action_chunk_size`.
- Each frame is encoded once during rollout; the policy/value input is a
  *dilated* stack of `frame_stack` latents spaced `frame_stride` steps apart
  (`LatentHistory`). The PPO update runs on stored latents — the frozen ViT
  never appears in the optimization loop.
- Parallel envs are managed as a plain Python list so time-limit truncation
  bootstrapping is explicit: on truncation the value of the genuine terminal
  observation is bootstrapped into the reward; on success it is not.
- The agent *contract* (`frame_stack`/`frame_stride`/`action_chunk_size`/
  `latent_dim`/`hidden_dim`/`action_dim`) is read from the BC `_stats.pth` and
  saved into every checkpoint; explicit CLI flags win.

## Usage

Requires the LeWM object checkpoint in the swm cache
(`python -m scripts.download_lewm_checkpoint`) and a trained BC checkpoint.
Run from the project root:

```bash
# tiny end-to-end smoke run
python -m src.ppo.train --smoke

# training run (flags accept dash or underscore spellings)
python src/ppo/train.py \
  --bc_checkpoint checkpoints/trained_policies/pusht_latent_bc.pth \
  --bc_stats checkpoints/trained_policies/pusht_latent_bc_stats.pth \
  --fixed_target --eval_interval 10 \
  --total_timesteps 1000000 --num_envs 8 --num_chunks 64

# evaluate + record videos
python -m src.ppo.evaluate \
  --checkpoint runs/<exp_name>__seed<seed>/<timestamp>/best.pt \
  --episodes 20 --video
```

Checkpoints (`latest.pt`, `best.pt`, `second_best.pt`, `final.pt`) and videos
are written under `runs/<exp_name>__seed<seed>/<timestamp>/`. With
`--eval_interval N > 0`, `best.pt` is selected by deterministic held-out
success (fixed seeds), matching `evaluate.py`.

## Logged metrics

`success_rate`, `final_distance`, `episodic_return`, `episodic_length`,
`eval/heldout_success`, plus PPO diagnostics (`approx_kl`, `clipfrac`,
`explained_variance`).
