# Offline RL with LeWM

## LeWorldModel Probing + Decoder Training

## BC

## PPO 

To train a latent PPO policy, you can use the following command:

```bash
python src/ppo/train_latent.py \
  --bc_checkpoint checkpoints/trained_policies/pusht_latent_bc.pth \
  --bc_stats checkpoints/trained_policies/pusht_latent_bc_stats.pth \
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

To evaluate the trained latent PPO policy, you can use the following command:

```bash
python -m src.ppo.evaluate_latent \    
  --checkpoint <CHECKPOINT_PATH> \
  --episodes 50 --video
```
