#!/usr/bin/env bash
#
# RQ1 training campaign: real-env PPO and dream PPO, one run per seed.
#
# These are the commands that produced the reference numbers (dream ~0.79, real
# ~0.89-0.92 at seed 1), with one addition on the real-PPO side:
#
#   --snapshot_interval 10
#
# Snapshots are what make the interaction-budget curve possible after the fact.
# Each one is named snapshot_step<env_steps>_it<iteration>.pt, so a checkpoint
# can be placed on the "success vs env steps consumed" axis without re-training.
# The dream command already passes it. Nothing else about the runs changes, so
# the seeds still reproduce the reference numbers.
#
# Usage:
#   bash scripts/rq1/run_campaign.sh              # seeds 1 2 3, both agents
#   bash scripts/rq1/run_campaign.sh --seeds "1 2 3 4 5"
#   bash scripts/rq1/run_campaign.sh --only dream
#
# Runs are sequential: both agents need the GPU and the frozen ViT, and
# interleaving them makes the steps/s numbers meaningless.

set -euo pipefail

SEEDS="1 2 3"
ONLY="both"
CONDA_ENV="dl_lab"
TOTAL_TIMESTEPS=1000000
PUSH_TO_HF=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;          # real | dream | both
    --env) CONDA_ENV="$2"; shift 2 ;;
    --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --push-to-hf) PUSH_TO_HF=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Optional flag arrays below are expanded as ${arr[@]+"${arr[@]}"} rather than
# "${arr[@]}": macOS ships bash 3.2, where expanding an *empty* array trips
# `set -u` with "unbound variable". The guard yields nothing when empty and
# preserves quoting when not.

BC_CHECKPOINT="hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth"
BC_STATS="hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth"

run_real() {
  local seed="$1"
  echo "=== real-env PPO | seed ${seed} ==="
  local hf_args=()
  if [[ "$PUSH_TO_HF" == "1" ]]; then
    hf_args=(--push_to_hf --hf_repo_id offline-rl-with-le-wm/ppo --hf_path_prefix "real_env_dense_seed${seed}")
  fi
  conda run --no-capture-output -n "$CONDA_ENV" \
  python src/ppo/train.py \
    --exp_name latent_ppo_pusht_real_dense_bc_v2 \
    --seed "$seed" \
    --bc_checkpoint "$BC_CHECKPOINT" \
    --bc_stats "$BC_STATS" \
    --hidden_dim 256 \
    --frame_stack 3 \
    --frame_stride 5 \
    --action_chunk_size 5 \
    --fixed_target \
    --block_start_near_goal \
    --block_start_radius 200.0 \
    --reward_mode dense \
    --eval_interval 10 \
    --eval_episodes 20 \
    --eval_seed 0 \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --num_envs 8 \
    --num_chunks 64 \
    --log_interval 10 \
    --save_interval 10 \
    --snapshot_interval 10 \
    ${hf_args[@]+"${hf_args[@]}"}
}

run_dream() {
  local seed="$1"
  echo "=== dream PPO | seed ${seed} ==="
  local hf_args=()
  if [[ "$PUSH_TO_HF" == "1" ]]; then
    hf_args=(--push-to-hf --hf-repo-id offline-rl-with-le-wm/ppo --hf-path-prefix "world_model_sparse_dense_correlation_seed${seed}")
  fi
  conda run --no-capture-output -n "$CONDA_ENV" \
  python -m src.ppo.train_lewm \
    --exp_name latent_ppo_pusht_lewm_sparse_dense_correlation \
    --seed "$seed" \
    --bc_checkpoint "$BC_CHECKPOINT" \
    --bc_stats "$BC_STATS" \
    --hidden_dim 256 \
    --frame_stack 3 \
    --frame_stride 5 \
    --action_chunk_size 5 \
    --wm_frameskip 5 \
    --fixed_target \
    --block_start_near_goal \
    --block_start_radius 200.0 \
    --reward_mode dense \
    --dense_reward_checkpoint models/probes/pusht_dense_reward_hf/dense_reward_classifier.pt \
    --dense_reward_mode potential \
    --dense_reward_coef 0.05 \
    --dense_reward_weights "1 0.75 0.4 0.1" \
    --dense_reward_clip 0.5 \
    --no-bc-penalty \
    --selection dream \
    --dream-eval-interval 10 \
    --dream-eval-episodes 96 \
    --dream-eval-seed 12345 \
    --record-real-eval \
    --eval-episodes 20 \
    --eval-seed 0 \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs 8 \
    --num-chunks 64 \
    --log-interval 10 \
    --save-interval 10 \
    --snapshot-interval 10 \
    ${hf_args[@]+"${hf_args[@]}"}
}

for seed in $SEEDS; do
  if [[ "$ONLY" == "real" || "$ONLY" == "both" ]]; then run_real "$seed"; fi
  if [[ "$ONLY" == "dream" || "$ONLY" == "both" ]]; then run_dream "$seed"; fi
done

echo
echo "Campaign done. Next:"
echo "  python -m scripts.rq1.evaluate_grid --seeds ${SEEDS}"
echo "  python -m scripts.rq1.budget_curve  --seeds ${SEEDS}"
echo "  python -m scripts.rq1.report"
