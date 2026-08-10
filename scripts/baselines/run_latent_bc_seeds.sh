#!/usr/bin/env bash
# The LeWM arm of the encoder-baseline comparison.
#
# The published campaign prior is a single training seed (42), which is not a
# sound reference for a multi-seed baseline. This retrains the *same* recipe with
# the *same* trainer at the encoder-baseline seeds, so the LeWM row and the
# oracle row carry matched seed counts.
#
# Nothing here republishes or replaces bc/pusht-bc-raw-cls: these are local
# comparison runs only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=(1 2 3)
DATA_PATH="data/expert_trajectories/pusht_expert_224.npz"
OUTPUT_ROOT="runs/baselines"
# Read from the campaign prior's own run config and weights, not from prose.
EPOCHS=1000
HIDDEN_DIM=256
LR=1e-3
BATCH_SIZE=64
EPISODES=150
EVAL_SEED=42
BLOCK_START_RADIUS=200
MAX_EPISODE_STEPS=300
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)    IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  echo "Usage: run_latent_bc_seeds.sh [--seeds '1 2 3'] [--epochs N] [--episodes N] [--dry-run]"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

if [[ ! -f "$DATA_PATH" ]]; then
    echo "missing expert dataset: $DATA_PATH" >&2
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    checkpoint="${OUTPUT_ROOT}/lewm/seed${seed}/pusht_latent_bc.pth"
    if [[ -e "$checkpoint" ]]; then
        echo "refusing to overwrite existing checkpoint: $checkpoint" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    run_dir="${OUTPUT_ROOT}/lewm/seed${seed}"
    checkpoint="${run_dir}/pusht_latent_bc.pth"
    echo "=== lewm latent BC | seed ${seed} ==="
    run mkdir -p "$run_dir"
    run python -u -m src.bc.train_bc_latent \
        --data_path "$DATA_PATH" \
        --checkpoint_path "$checkpoint" \
        --observation-resolution 224 \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --hidden_dim "$HIDDEN_DIM" \
        --frame_stack 3 \
        --frame_stride 5 \
        --action_chunk_size 5 \
        --seed "$seed" \
        --deterministic \
        --log_interval 50 \
        --save_interval 100

    run python -u -m src.evaluation.evaluate_pusht \
        --agent-type bc \
        --checkpoint "$checkpoint" \
        --block-start-radius "$BLOCK_START_RADIUS" \
        --episodes "$EPISODES" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --seed "$EVAL_SEED" \
        --output-root "${OUTPUT_ROOT}/evaluations" \
        --run-name "lewm_seed${seed}"
done

echo "LeWM latent BC seeds complete. Evaluations under ${OUTPUT_ROOT}/evaluations/"
