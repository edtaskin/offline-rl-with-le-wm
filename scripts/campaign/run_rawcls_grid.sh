#!/usr/bin/env bash
#
# One BC prior, one protocol, every PPO arm.
#
# Every run here starts from the published raw-CLS BC checkpoint
#
#   hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth
#
# so real-env and dream results sit in the same table without a "different
# prior" footnote. That checkpoint is 256-wide raw CLS, matching the
# architecture the PPO hyperparameters in runs/ppo_hpo were tuned for.
#
# Arms (see the reward-axis note below):
#
#   real_sparse    real env,  1.0 on success
#   real_dense     real env,  block-pose distance shaping
#   dream_sparse   LeWM,      1.0 on probe-declared success
#   dream_dense    LeWM,      sparse + learned time-to-success classifier
#   dream_success_potential
#                  LeWM,      objective_met probability potential shaping
#
# REWARD AXIS IS NOT SYMMETRIC. "dense" means block-pose shaping in the real env
# and the learned classifier in the dream; the dream's counterpart to real_dense
# is reward_mode=pose_dense, which needs the block_rel_objective regression probe
# under models/probes/pusht_lewm/. That probe is not published alongside the
# objective_met classifier, so the matched dense pair cannot be run yet -- train
# it with scripts/probes/train_state.py to unlock a --only dream_pose arm.
# Read the dense column as two different shaping schemes, not as one comparison.
#
# Selection: best.pt comes from real held-out success for the real arms and from
# imagined success for the dream arms (--selection dream), which is the RQ1
# contrast. Note the prior itself was chosen by real-env success, so a dream run
# started here is not interaction-free end to end; RQ1 runs should use
# pusht_bc_raw_cls_final.pth instead.
#
# Publishing: each run pushes available selection checkpoints (best.pt,
# second_best.pt, best_heldout_real.pt), final.pt and (dream) selection_log.jsonl
# under rawcls_bc_best/<arm>/seed<N>/ in the Hub repo. Destinations are checked
# for collisions before any training starts, so nothing existing is overwritten.
#
# Prerequisites on a fresh machine: the LeWM object checkpoint
# (python -m scripts.download_lewm_checkpoint) and .env with WANDB_API_KEY and
# HF_TOKEN. BC priors are pulled from the Hub automatically. The real-env arms need
# nothing else; the dream arms additionally need the expert h5 and the probes.
#
# Usage:
#   bash scripts/campaign/run_rawcls_grid.sh                      # 4 arms x seeds 1 2 3
#   bash scripts/campaign/run_rawcls_grid.sh --seeds "1"          # one seed
#   bash scripts/campaign/run_rawcls_grid.sh --only dream_sparse
#   bash scripts/campaign/run_rawcls_grid.sh --only "real_sparse real_dense"
#   bash scripts/campaign/run_rawcls_grid.sh --dry-run            # print commands only
#
# --exp-prefix names the run dirs, the W&B runs and (by default) the Hub folder, so
# a second prior can reuse this script without colliding with the first campaign.
# See scripts/campaign/run_legacy_prior_control.sh for that case.
#
# Runs are sequential: every arm wants the GPU and the frozen ViT.

set -euo pipefail

SEEDS="1 2 3"
ONLY="all"
DRY_RUN=0
NO_PUSH=0
NO_EVAL=0
TOTAL_TIMESTEPS=1000000
DREAM_EPISODE_STEPS=20
DREAM_EVAL_STEPS=20          # pinned, so the selection metric stays comparable
EVAL_EPISODES=50             # in-training held-out eval; 20 is too noisy to rank with
HF_REPO="offline-rl-with-le-wm/ppo"
EXP_PREFIX="rawcls"
ARM_SUFFIX=""
HF_NAMESPACE=""
WANDB_PROJECT="offline-rl-lewm"
BC_CKPT="hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth"
BC_STATS="hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best_stats.pth"
BRIDGE="deprojector"
DECODER_CHECKPOINT=""      # empty -> DreamConfig default (repo-local path)
SPARSE_REWARD_CHECKPOINT="hf://offline-rl-with-le-wm/sparse_reward_classifier"
DENSE_REWARD_CHECKPOINT="hf://offline-rl-with-le-wm/dense_reward_classifier/dense_reward_classifier.pt"
DENSE_REWARD_COEF="0.05"
DENSE_REWARD_CLIP="0.5"
DENSE_REWARD_WEIGHTS="1 0.75 0.4 0.1"
BC_PENALTY_COEF="0.05"
SUCCESS_POTENTIAL_COEF="1.0"
SUCCESS_POTENTIAL_CLIP="1.0"
SUCCESS_POTENTIAL_POSITIVE_ONLY=0
ANNEAL_LR_ARG=()
EVAL_VARIANTS=(best second_best best_heldout_real final)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --dream-episode-steps) DREAM_EPISODE_STEPS="$2"; shift 2 ;;
    --dream-eval-steps) DREAM_EVAL_STEPS="$2"; shift 2 ;;
    --bridge) BRIDGE="$2"; shift 2 ;;
    --decoder-checkpoint) DECODER_CHECKPOINT="$2"; shift 2 ;;
    --sparse-reward-checkpoint) SPARSE_REWARD_CHECKPOINT="$2"; shift 2 ;;
    --dense-reward-checkpoint) DENSE_REWARD_CHECKPOINT="$2"; shift 2 ;;
    --dense-reward-coef) DENSE_REWARD_COEF="$2"; shift 2 ;;
    --dense-reward-clip) DENSE_REWARD_CLIP="$2"; shift 2 ;;
    --dense-reward-weights) DENSE_REWARD_WEIGHTS="$2"; shift 2 ;;
    --bc-penalty-coef) BC_PENALTY_COEF="$2"; shift 2 ;;
    --success-potential-coef) SUCCESS_POTENTIAL_COEF="$2"; shift 2 ;;
    --success-potential-clip) SUCCESS_POTENTIAL_CLIP="$2"; shift 2 ;;
    --success-potential-positive-only) SUCCESS_POTENTIAL_POSITIVE_ONLY=1; shift ;;
    --anneal-lr|--anneal_lr) ANNEAL_LR_ARG=(--anneal_lr); shift ;;
    --no-anneal-lr|--no-anneal_lr) ANNEAL_LR_ARG=(--no-anneal_lr); shift ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --hf-namespace) HF_NAMESPACE="$2"; shift 2 ;;
    --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
    --arm-suffix) ARM_SUFFIX="$2"; shift 2 ;;
    --bc-checkpoint) BC_CKPT="$2"; shift 2 ;;
    --bc-stats) BC_STATS="$2"; shift 2 ;;
    --no-push) NO_PUSH=1; shift ;;
    --no-eval) NO_EVAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,44p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Namespace defaults to the experiment prefix, so two priors never share a
# Hub folder or a runs/ directory.
HF_NAMESPACE="${HF_NAMESPACE:-${EXP_PREFIX}_bc_best}"

ARMS="real_sparse real_dense dream_sparse dream_dense"
if [[ "$ONLY" != "all" ]]; then ARMS="$ONLY"; fi

# ---- preflight: the LeWM encoder is not downloaded on demand -----------------
if [[ "$DRY_RUN" -eq 0 ]]; then
  python - <<'PYCHECK' || { echo "aborting: LeWM encoder checkpoint missing" >&2; exit 1; }
import sys
sys.path.insert(0, ".")
from src.representations.lewm import default_lewm_checkpoint_path
path = default_lewm_checkpoint_path()
if not path.exists():
    print(f"missing LeWM object checkpoint: {path}", file=sys.stderr)
    print("run: python -m scripts.download_lewm_checkpoint", file=sys.stderr)
    raise SystemExit(1)
print(f"== LeWM encoder ok: {path}")
PYCHECK
fi

# ---- preflight: dream arms need the expert h5 and the success probe -----------
if [[ "$DRY_RUN" -eq 0 && "$ARMS" == *dream* ]]; then
  python - <<'PYCHECK' || { echo "aborting: dream prerequisites missing" >&2; exit 1; }
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.ppo.train_lewm import DreamConfig, repo_path
cfg = DreamConfig()
missing = []
dataset = repo_path(cfg.dataset_path)
if not dataset.is_file():
    missing.append(f"expert dataset {dataset} (~46 GB; copy it or re-export it)")
probe_dir = repo_path(cfg.probe_dir)
if not cfg.sparse_reward_checkpoint.startswith("hf://") and not Path(cfg.sparse_reward_checkpoint).is_file():
    missing.append(
        f"sparse reward classifier {cfg.sparse_reward_checkpoint} "
        "(or pass --sparse-reward-checkpoint hf://.../objective_met/mlp_probe.pt)"
    )
for item in missing:
    print(f"missing: {item}", file=sys.stderr)
raise SystemExit(1 if missing else 0)
PYCHECK
fi

# ---- preflight: refuse to start if any destination is already taken ----------
if [[ "$NO_PUSH" -eq 0 ]]; then
  PREFIXES=()
  for arm in $ARMS; do
    for seed in $SEEDS; do
      PREFIXES+=("${HF_NAMESPACE}/${arm}${ARM_SUFFIX}/seed${seed}")
    done
  done
  echo "== checking ${#PREFIXES[@]} Hub destination(s) in ${HF_REPO}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    python -m scripts.campaign.check_hf_prefix --repo-id "$HF_REPO" --prefix "${PREFIXES[@]}"
  fi
fi

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

COMMON=(
  --bc_checkpoint "$BC_CKPT"
  --bc_stats "$BC_STATS"
  --total_timesteps "$TOTAL_TIMESTEPS"
  --num_envs 8 --num_chunks 64
  --fixed_target --block_start_near_goal --block_start_radius 200
  --observation-resolution 224
  --snapshot_interval 10
  --track --wandb_project "$WANDB_PROJECT"
  ${ANNEAL_LR_ARG[@]+"${ANNEAL_LR_ARG[@]}"}
)

for seed in $SEEDS; do
  for arm in $ARMS; do
    exp="${EXP_PREFIX}_${arm}${ARM_SUFFIX}"
    prefix="${HF_NAMESPACE}/${arm}${ARM_SUFFIX}/seed${seed}"
    push=()
    if [[ "$NO_PUSH" -eq 0 ]]; then
      push=(--push_to_hf --hf_repo_id "$HF_REPO" --hf_path_prefix "$prefix")
    fi

    echo ""
    echo "=============================================================="
    echo "== ${arm} | seed ${seed} | -> ${prefix}"
    echo "=============================================================="

    case "$arm" in
      real_sparse|real_dense)
        reward="${arm#real_}"
        run python -m src.ppo.train \
          --exp_name "$exp" --seed "$seed" \
          --reward_mode "$reward" \
          --eval_interval 10 --eval_episodes "$EVAL_EPISODES" \
          "${COMMON[@]}" ${push[@]+"${push[@]}"}
        ;;
      dream_sparse|dream_dense|dream_success_potential)
        reward="${arm#dream_}"
        decoder_arg=()
        if [[ -n "$DECODER_CHECKPOINT" ]]; then
          decoder_arg=(--decoder_checkpoint "$DECODER_CHECKPOINT")
        fi
        dense_arg=()
        if [[ "$reward" == "dense" ]]; then
          dense_arg=(
            --dense_reward_checkpoint "$DENSE_REWARD_CHECKPOINT"
            --dense_reward_coef "$DENSE_REWARD_COEF"
            --dense_reward_clip "$DENSE_REWARD_CLIP"
            --dense_reward_weights "$DENSE_REWARD_WEIGHTS"
            --dense_reward_mode potential
          )
        fi
        success_potential_arg=()
        if [[ "$reward" == "success_potential" ]]; then
          success_potential_arg=(
            --success_potential_coef "$SUCCESS_POTENTIAL_COEF"
            --success_potential_clip "$SUCCESS_POTENTIAL_CLIP"
          )
          if [[ "$SUCCESS_POTENTIAL_POSITIVE_ONLY" -eq 1 ]]; then
            success_potential_arg+=(--success_potential_positive_only)
          fi
        fi
        run python -m src.ppo.train_lewm \
          --exp_name "$exp" --seed "$seed" \
          --bridge "$BRIDGE" \
          --reward_mode "$reward" \
          --sparse_reward_checkpoint "$SPARSE_REWARD_CHECKPOINT" \
          --bc_penalty --bc_penalty_coef "$BC_PENALTY_COEF" \
          --dream_episode_steps "$DREAM_EPISODE_STEPS" \
          --dream_eval_steps "$DREAM_EVAL_STEPS" \
          --selection dream --dream_eval_interval 10 --dream_eval_episodes 96 \
          --record_real_eval --eval_interval 10 --eval_episodes "$EVAL_EPISODES" \
          ${decoder_arg[@]+"${decoder_arg[@]}"} \
          ${dense_arg[@]+"${dense_arg[@]}"} \
          ${success_potential_arg[@]+"${success_potential_arg[@]}"} \
          "${COMMON[@]}" ${push[@]+"${push[@]}"}
        ;;
      *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac

    # ---- evaluation: same protocol for every arm, both checkpoints ----------
    if [[ "$NO_EVAL" -eq 1 ]]; then continue; fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "+ (eval) ${exp}__seed${seed} ${EVAL_VARIANTS[*]}"
      continue
    fi
    run_dir="$(ls -d runs/${exp}__seed${seed}/*/ | tail -1)"
    for variant in "${EVAL_VARIANTS[@]}"; do
      [[ -f "${run_dir}${variant}.pt" ]] || continue
      run env SDL_VIDEODRIVER=dummy python -m src.evaluation.evaluate_pusht \
        --agent-type ppo \
        --checkpoint "${run_dir}${variant}.pt" \
        --observation-resolution 224 --training-observation-resolution 224 \
        --block-start-radius 200 \
        --episodes 150 --seed 42 --max-episode-steps 300 \
        --execution-mode open-loop \
        --run-name "${EXP_PREFIX}-${arm}${ARM_SUFFIX}-seed${seed}-${variant}" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "${EXP_PREFIX}-${arm}${ARM_SUFFIX}-seed${seed}-${variant}-eval"
    done
  done
done

echo ""
echo "campaign complete: arms [${ARMS}] x seeds [${SEEDS}]"
