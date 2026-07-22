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
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
  --data_path data/expert_trajectories/pusht_expert_224.npz \
  --seed 42 \
  --num_workers 0 \
  --deterministic \
  --log_interval 10 \
  --save_interval 100 \
  --push_to_hf \
  --hf_repo_id offline-rl-with-le-wm/behavioral-cloning
```

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
and does not affect policy inputs.

Evaluate BC on the fixed-target task. Passing `--block-start-radius 200`
matches the PPO training start distribution; omit it for unrestricted block
starts around the same fixed target.

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type bc \
  --checkpoint hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth \
  --stats hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth \
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
  --bc_checkpoint hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth \
  --bc_stats hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
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
  --block-start-radius 200 \
  --episodes 50 \
  --max-episode-steps 300 \
  --seed 42 \
  --repeats 3 \
  --video
```
