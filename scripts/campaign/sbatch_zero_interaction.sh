#!/usr/bin/env bash
#
# Zero-interaction dream campaign, one seed per array task.
#
# Trains dream_dense and dream_sparse from the projected BC prior with
# --interaction-free, so PPO consumes no environment steps: no diagnostic
# real-env eval during training, and best.pt selected from imagined success.
# The prior is the *final* BC checkpoint, not best -- best was chosen by
# real-env success, which would put 740k env steps upstream of an otherwise
# interaction-free run. The post-training canonical evaluation still runs; it
# measures finished checkpoints and feeds nothing back into training.
#
# Submit:
#   mkdir -p logs/slurm
#   sbatch scripts/campaign/sbatch_zero_interaction.sh
#
# Seeds come from the array range: --array=1-3 runs seeds 1, 2 and 3 in
# parallel. Override at submit time without editing the file:
#   sbatch --array=1-5 scripts/campaign/sbatch_zero_interaction.sh
#
#SBATCH --job-name=lewm-zero
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.out
#
# ---------------------------------------------------------------------------
# CLUSTER-SPECIFIC: uncomment and set for your partition/account before the
# first submission. Left unset, the job lands on the cluster's default queue,
# which may not have a GPU.
# ---------------------------------------------------------------------------
##SBATCH --partition=
##SBATCH --account=

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-dl-lab-project}"

# The LeWM object checkpoint is found via stable_worldmodel's get_cache_dir(),
# which reads STABLEWM_HOME and silently falls back to its own default when the
# variable is missing. Batch shells do not source ~/.bashrc, so without this
# line the preflight fails with "missing LeWM object checkpoint" on a machine
# where the file is plainly present.
export STABLEWM_HOME="${STABLEWM_HOME:-/home/edtaskin/uni-freiburg/SoSe26/Deep-Learning-Lab/DL_Project/stable_wm_data}"

# Headless compute node: pygame/pymunk rendering during the post-training eval.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEED="${SLURM_ARRAY_TASK_ID:-1}"
BC_DIR="hf://offline-rl-with-le-wm/bc/latent-bc/seed42"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- preflight: fail in seconds, not after the queue wait ------------------
LEWM_CKPT="${STABLEWM_HOME}/checkpoints/pusht/lewm_object.ckpt"
if [[ ! -f "$LEWM_CKPT" ]]; then
  echo "missing LeWM object checkpoint: $LEWM_CKPT" >&2
  echo "set STABLEWM_HOME above, or run: python -m scripts.download_lewm_checkpoint" >&2
  exit 1
fi
if [[ ! -f "le-wm/models/datasets/pusht_expert_train.h5" ]]; then
  echo "missing expert h5 (44 GB); the dream world samples context windows from it" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "missing .env with WANDB_API_KEY and HF_TOKEN" >&2
  exit 1
fi

echo "== node $(hostname) | seed ${SEED} | env ${CONDA_ENV} | $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/campaign/run_rawcls_grid.sh \
  --interaction-free \
  --only "dream_dense dream_sparse" \
  --seeds "$SEED" \
  --exp-prefix projected_zero \
  --hf-namespace projected_zero_bc_final \
  --bc-checkpoint "${BC_DIR}/pusht_latent_bc_final.pth" \
  --bc-stats "${BC_DIR}/pusht_latent_bc_final_stats.pth"

echo "== done | seed ${SEED} | $(date -Is)"
