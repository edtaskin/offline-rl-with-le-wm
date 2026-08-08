#!/usr/bin/env bash
# Encoder baselines for the BC prior: train and score one representation per arm
# under the canonical evaluation protocol.
#
# These are diagnostics for "is the frozen LeWM CLS token the binding constraint
# on BC?". They are deliberately never wired into scripts/campaign/ — the PPO
# arms already carry a prior confound and must keep exactly one prior family.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENCODERS=("state")
# Training seeds match the PPO campaign convention. The evaluation master seed
# below is separate and stays at 42: it is the canonical protocol, not an arm.
SEEDS=(1 2 3)
DATA_PATH="data/expert_trajectories/pusht_expert_224.npz"
OUTPUT_ROOT="runs/baselines"
# Matched to the campaign prior (bc/pusht-bc-raw-cls), read from that run's own
# config and weights rather than from prose: 256-wide head, 1000 epochs, lr 1e-3,
# batch 64, same temporal contract. Only the representation differs.
EPOCHS=1000
HIDDEN_DIM=256
LR=1e-3
BATCH_SIZE=64
EPISODES=150
EVAL_SEED=42
BLOCK_START_RADIUS=200
MAX_EPISODE_STEPS=300
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_encoder_baselines.sh [options]

  --encoders <list>   space-separated encoder arms (default: state)
  --seeds <list>      space-separated training seeds (default: 1 2 3)
  --epochs <n>        training epochs per run (default: 1000)
  --episodes <n>      canonical evaluation episodes (default: 150)
  --dry-run           print the commands without running them
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoders) IFS=' ' read -r -a ENCODERS <<< "$2"; shift 2 ;;
        --seeds)    IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
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
    echo "run scripts/regenerate_pusht_expert.py first" >&2
    exit 1
fi

# Preflight every destination before the first run, so a late collision cannot
# waste the whole sweep.
for encoder in "${ENCODERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        checkpoint="${OUTPUT_ROOT}/${encoder}/seed${seed}/pusht_${encoder}_bc.pth"
        if [[ -e "$checkpoint" ]]; then
            echo "refusing to overwrite existing checkpoint: $checkpoint" >&2
            exit 1
        fi
    done
done

for encoder in "${ENCODERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${encoder}/seed${seed}"
        checkpoint="${run_dir}/pusht_${encoder}_bc.pth"
        echo "=== ${encoder} baseline | seed ${seed} ==="
        run mkdir -p "$run_dir"
        # -u keeps progress visible when the sweep is redirected to a log.
        run python -u -m src.bc.train_bc_baseline \
            --encoder "$encoder" \
            --data_path "$DATA_PATH" \
            --checkpoint_path "$checkpoint" \
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
            --agent-type "bc-${encoder}" \
            --checkpoint "$checkpoint" \
            --block-start-radius "$BLOCK_START_RADIUS" \
            --episodes "$EPISODES" \
            --max-episode-steps "$MAX_EPISODE_STEPS" \
            --seed "$EVAL_SEED" \
            --output-root "${OUTPUT_ROOT}/evaluations" \
            --run-name "${encoder}_seed${seed}"
    done
done

echo "Encoder baselines complete. Evaluations under ${OUTPUT_ROOT}/evaluations/"
