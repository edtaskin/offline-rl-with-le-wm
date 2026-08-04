# Offline RL with LeWM

Trained policy artifacts are published through Hugging Face and kept out of
Git. `hf://<owner>/<repo>/<filename>` references check the requested Hub
revision on each process start and reuse the Hugging Face cache when its commit
has not changed. Legacy paths under `checkpoints/` resolve to registered Hub
artifacts and are never read from the local filesystem.

## LeWorldModel Probing + Decoder Training

## BC

Generate the LeWM-resolution expert dataset once after downloading the original
PushT demonstrations. This restores each recorded simulator state and renders
it directly at 224x224; it does not interpolate the stored 96x96 pixels.

```bash
python scripts/regenerate_pusht_expert.py \
  --dataset data/expert_trajectories/pusht_expert.npz \
  --output-dataset data/expert_trajectories/pusht_expert_224.npz \
  --verify-n 100
```

The generator uses a disk-backed image buffer. During the final save, allow
space for both that temporary buffer and the output dataset. Add `--compressed`
if disk space matters more than generation and load time.

Train the latent BC policy with the configuration used for the current
checkpoint:

```bash
python -m src.bc.train_bc_latent \
  --data_path data/expert_trajectories/pusht_expert_224.npz \
  --checkpoint_path runs/bc/pusht_latent_bc.pth \
  --observation-resolution 224 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
  --seed 42 \
  --num_workers 0 \
  --deterministic \
  --log_interval 10 \
  --save_interval 100 \
  --eval_interval 50 \
  --eval_episodes 50 \
  --eval_repeats 3 \
  --push_to_hf \
  --hf_repo_id offline-rl-with-le-wm/behavioral-cloning
```

The trainer runs the canonical fixed-target PushT evaluator every 50 epochs and
once at the final epoch if needed. It writes `pusht_latent_bc_best.pth`, selected
by aggregate evaluation success rate, and `pusht_latent_bc_final.pth`, together
with matching `_stats.pth` files and an evaluation-history JSON file. When Hub
uploading is enabled, all artifacts from one training run are placed in one
folder. The folder defaults to `--wandb_run_name` when set, otherwise to the
checkpoint stem; set `--hf_path_prefix` to choose it explicitly.

Each evaluation creates a timestamped directory under `runs/evaluations/`.
By default, it runs 3 repeats of 50 episodes. `--seed` sets the first repeat's
seed; later repeats use deterministic, non-overlapping seed ranges. The pooled
summary and each repeat's summary are saved in `metrics.json`; with `--video`,
episode videos are saved in repeat-specific directories under `videos/`. Use
`--output-root` to change the parent directory and `--run-name` to append a
readable label. Set `--repeats 1` for a single evaluation.

Policy observations render at 224x224 by default. This resolution is stored in
`metrics.json` as `config.observation_resolution`; pass
`--observation-resolution 96` to reproduce the earlier low-resolution
evaluation protocol. `--video-resolution` controls only saved-video scaling
and does not affect policy inputs. New BC stats and PPO checkpoints record their
native training observation resolution, and evaluation rejects a mismatch by
default. For older artifacts that predate this metadata, provide
`--training-observation-resolution`. Use `--allow-resolution-mismatch` only for
an intentional resolution-transfer experiment.

Evaluate BC on the fixed-target task. Passing `--block-start-radius 200`
matches the PPO training start distribution; omit it for unrestricted block
starts around the same fixed target.

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type bc \
  --checkpoint hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc/pusht_latent_bc_best.pth \
  --stats hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc/pusht_latent_bc_best_stats.pth \
  --training-observation-resolution 224 \
  --block-start-radius 200 \
  --episodes 50 \
  --max-episode-steps 300 \
  --video \
  --seed 42 \
  --repeats 3
```

## PPO 

To train a latent PPO policy, you can use the following command:

```bash
python src/ppo/train.py \
  --bc_checkpoint hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc/pusht_latent_bc_best.pth \
  --bc_stats hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc/pusht_latent_bc_best_stats.pth \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
  --observation-resolution 224 \
  --fixed_target \
  --log_interval 10 \
  --save_interval 10 \
  --eval_interval 10 \
  --total_timesteps 1000000 --num_envs 8 --num_chunks 64 \
  --push_to_hf \
  --hf_repo_id offline-rl-with-le-wm/ppo
```

PPO uses the same environment-owned, fixed-target evaluator as BC. The
checkpoint and agent type select the PPO adapter.

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type ppo \
  --checkpoint hf://offline-rl-with-le-wm/ppo/best.pt \
  --observation-resolution 96 \
  --training-observation-resolution 96 \
  --block-start-radius 200 \
  --episodes 50 \
  --max-episode-steps 300 \
  --seed 42 \
  --repeats 3 \
  --video
```

## RQ1: can policy improvement happen inside the world model?

`src/ppo/train_lewm.py` runs the same chunk-level latent PPO as `src/ppo/ppo.py`,
but rolls out inside LeWM instead of the simulator: expert data supplies episode
starts, the frozen world model supplies dynamics, the decoder bridges back to the
policy's latent space, and a frozen probe supplies reward and success. No
`env.step` is called.

Two things make "zero environment interaction" a checkable claim rather than a
description:

* **Interaction is counted.** Every checkpoint records `env_steps_consumed`
  (`train_env_steps + eval_env_steps`). Held-out evaluation counts, because
  picking `best.pt` by real-env success is interaction even though it trains
  nothing. Dream runs report training steps as 0 by construction.
* **Selection can be interaction-free.** `--selection dream` ranks checkpoints by
  imagined success on expert episodes held out at *episode* granularity
  (`--dream_val_fraction`), so `best.pt` is chosen without a simulator. Use
  `--selection real` for the interaction-paying contrast, and `final.pt` as the
  no-selection control.

Train one dream agent whose entire interaction budget is zero:

```bash
python -m src.ppo.train_lewm \
  --exp_name rq1_dream_ppo --seed 1 \
  --fixed_target --reward_mode sparse \
  --block_start_near_goal --block_start_radius 200 \
  --total_timesteps 1000000 --num_envs 8 --num_chunks 64 \
  --eval_interval 0 \
  --selection dream --dream_eval_interval 25 --dream_eval_episodes 96 \
  --snapshot_interval 25
```

`--snapshot_interval` keeps permanent `snapshot_step<env_steps>_it<iteration>.pt`
checkpoints; unlike `latest.pt` they are never overwritten, which is what makes an
interaction-budget curve possible after the fact. It applies to `src/ppo/train.py`
too.

### Running the RQ1 campaign

Three matched seeds per agent, one evaluation protocol, then the table, the
success figure and the interaction-budget curve:

```bash
bash scripts/rq1/run_campaign.sh --seeds "1 2 3"   # trains real-env PPO and dream PPO
python -m scripts.rq1.evaluate_grid --seeds 1 2 3  # canonical eval, 3 repeats x 50 episodes
python -m scripts.rq1.budget_curve  --seeds 1 2 3  # real-PPO snapshots vs env steps
python -m scripts.rq1.report                       # table.md + figures + summary.json
```

`run_campaign.sh` is the reference training command with `--snapshot_interval 10`
added on the real-PPO side — without step-tagged snapshots there is no budget
curve to draw. Training is not resumable, but evaluation is: both evaluation
scripts append to `runs/rq1/*.jsonl` and skip checkpoints already scored under the
same protocol, so an interrupted campaign continues where it stopped.

`evaluate_grid.py` scores three checkpoints per PPO run, which is what makes the
selection story explicit:

| variant | what selected it | interaction it required |
|---|---|---|
| `final` | nothing — the last checkpoint | training only (0 for dream PPO) |
| `best` | the run's own `--selection` rule | 0 for `--selection dream` |
| `best_real_sel` | real held-out success, reconstructed post-hoc from `selection_log.jsonl` | training + every eval step up to that checkpoint |

`report.py` reports **required** env steps, not merely consumed ones: a dream run
launched with `--record-real-eval` also spends simulator steps, but those are RQ2
instrumentation that never feeds selection, so they are listed separately as
diagnostic rather than charged to the claim.

## RQ2: is the imagined reward trustworthy?

The dream reward *and* the dream episode's termination both come from one frozen
probe reading a latent the simulator never corrects. That is the textbook setup
for model exploitation, and the run is already instrumented to measure it — three
scripts, one per piece of evidence:

```bash
# 1. optimism: imagined vs real held-out success, same weights, same x-axis
python -m scripts.rq2.optimism_gap --seeds 1 2 3

# 2. trust vs depth: probe false-positive rate over a 5 -> 100 env-step horizon
python -m scripts.rq2.probe_horizon --seeds 1 2 3

# 3. the frames PPO believed were successes, checked against the simulator
python -m scripts.rq2.dream_success_gallery --seed 1 --episodes 96 --verify-in-sim
```

* **`optimism_gap.py`** needs `--record-real-eval` on the dream run: the trainer
  then writes both metrics into `selection_log.jsonl` at the same cadence for the
  same weights. Reports the level gap, the rank agreement (Spearman ρ), and the
  real success given up by selecting on the imagined metric.
* **`probe_horizon.py`** resolves the `objective_met` classifier exactly the way
  `LeWMDreamWorld` does and pins `scripts/probes/probe_rollouts_pusht.py` to that
  file, so the curve describes the probe PPO actually optimized against. It plots
  the imagined false-positive rate against the *encoded ground-truth* rate, which
  separates probe error from world-model drift, and marks the horizon the agent
  actually lives in.
* **`dream_success_gallery.py`** rolls the trained policy in the dream, catches
  probe-declared successes, and decodes the latent the probe made that call on.
  `--verify-in-sim` replays the same action chunks in the simulator from the same
  dataset start state, turning "these look wrong" into a hallucination rate. The
  replay is open-loop by design — the question is whether the *imagined
  trajectory* was real, not how good the policy is (that is RQ1's job) — and its
  `mean_anchor_pixel_error` reports how exactly the simulator reproduced the
  dream's start frame, so an unreliable verdict is visible rather than silent.
