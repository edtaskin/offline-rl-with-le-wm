#!/usr/bin/env bash
#
# BC-checkpoint selection sweep: one dream-PPO run per (BC checkpoint x seed).
#
# The question this answers is "which stage of BC training is the best *RL
# initialization*", which is not the same question as "which BC is the best
# policy". In the default configuration the BC enters PPO only through
# build_latent_agent (src/ppo/agent.py) -- it seeds the actor's mean network and
# nothing else, since exploration std comes from --init_log_std and the BC
# penalty is off. So what varies across these runs is the optimization starting
# point and the region of latent space the imagined rollouts open in.
#
# Two properties of this script are what make its answer trustworthy:
#
#   1. MULTIPLE PPO SEEDS PER CHECKPOINT (--seeds, default 3). With one seed
#      each, argmax over final success mostly selects the luckiest seed; the
#      across-seed spread here is comparable to the effect being measured.
#
#   2. A FIXED REWARD CONFIGURATION (see the REWARD_ARGS block). Selecting the
#      BC under one reward probe and then benchmarking probes against each other
#      co-adapts the initialization to whichever probe was used for selection.
#      Selection therefore runs under the canonical dense reward and stays there.
#      If you later vary the reward probe, do NOT re-point this at the probe
#      under test.
#
# Everything else is copied verbatim from run_dream() in scripts/rq1/run_campaign.sh
# so that a checkpoint chosen here is chosen under the pipeline it will be used in.
#
# The script then evaluates what it trained and prints the recommendation, so one
# invocation answers the question end to end (--no-eval stops after training).
# Two consequences of chaining are worth knowing:
#
#   * A failed PPO run no longer aborts the sweep. Losing five finished cells
#     because the sixth ran out of memory is the wrong trade when the grid takes
#     days; failures are collected, the surviving cells are still evaluated and
#     reported, and the script exits non-zero at the very end.
#   * --seed-role tags this invocation's seeds as the ones the choice is MADE on
#     ("select", the default) or the fresh ones it is RE-MEASURED on ("confirm").
#     evaluate_sweep.py keeps the two apart because the selection mean is
#     optimistically biased, so a confirmation pass must say so.
#
# Publishing is never automatic: the report prints the hf:// publish command once
# a confirmation pass exists. See --push-to-hf in report.py.
#
# Usage: ONE command does the whole thing.
#
#   # Prerequisite (once): a long BC run that keeps its intermediate epochs.
#   python src/bc/train_bc_latent.py --epochs 120 --save_interval 20 \
#     --checkpoint_path runs/bc_long/pusht_bc.pth
#
#   bash scripts/bc_selection/run_sweep.sh --bc-dir runs/bc_long --wandb
#
# That runs both phases back to back without waiting on you:
#
#   SELECTION    every checkpoint x --seeds (default "1 2 3"), then evaluate,
#                then report. The report names a winner by the plateau rule.
#   CONFIRMATION the winner alone, re-trained on --confirm-seeds (default: the
#                selection seeds + 10), then evaluated and reported again. This
#                is chained rather than left to you because the choice is made
#                by a deterministic rule, not a human judgement -- there is
#                nothing to stop and decide. The second report is the one to
#                quote: the selection mean is the max over checkpoints of a
#                noisy quantity and is biased upward.
#
# The confirmation pass is skipped automatically if any selection run failed,
# since a recommendation drawn from a partial grid is not worth confirming.
#
# Publishing stays a separate, deliberate command -- it writes to a public Hub
# repo, so it is never a side effect of a sweep:
#
#   python -m scripts.bc_selection.report --wandb --push-to-hf
#
# Useful variations:
#   --no-confirm     stop after selection (inspect the table before committing
#                    GPU time to the confirmation pass)
#   --no-eval        train only; evaluate and report later
#   --seed-role confirm   treat --seeds as confirmation seeds for a checkpoint
#                    chosen earlier; implies no further chaining
#
# Runs are sequential: each needs the GPU and the frozen ViT.

set -euo pipefail

SEEDS="1 2 3"
CONFIRM_SEEDS=""
SEED_ROLE="select"
BC_DIR=""
BC_CHECKPOINTS=""
BC_STATS=""
EXP_PREFIX="bcsel_dream"
OUTPUT_ROOT="runs/bc_selection"
CONDA_ENV="dl_lab"
TOTAL_TIMESTEPS=1000000
DRY_RUN=0
RUN_EVAL=1
AUTO_CONFIRM=1
WANDB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bc-dir) BC_DIR="$2"; shift 2 ;;                    # glob <dir>/*.pth
    --bc-checkpoints) BC_CHECKPOINTS="$2"; shift 2 ;;    # explicit space-separated list
    --bc-stats) BC_STATS="$2"; shift 2 ;;                # defaults to <dir>/*_stats.pth
    --seeds) SEEDS="$2"; shift 2 ;;
    --confirm-seeds) CONFIRM_SEEDS="$2"; shift 2 ;;      # defaults to --seeds + 10
    --seed-role) SEED_ROLE="$2"; shift 2 ;;              # select | confirm
    --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --no-eval) RUN_EVAL=0; shift ;;                      # train only; evaluate later
    --no-confirm) AUTO_CONFIRM=0; shift ;;               # stop after selection
    --wandb) WANDB=1; shift ;;                           # forwarded to report.py
    --dry-run) DRY_RUN=1; shift ;;
    # Print the header block itself rather than a hard-coded line range, which
    # silently starts truncating the moment anything is added above.
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SEED_ROLE" != "select" && "$SEED_ROLE" != "confirm" ]]; then
  echo "--seed-role must be 'select' or 'confirm', got '${SEED_ROLE}'." >&2
  exit 2
fi

# Default confirmation seeds are the selection seeds shifted by 10, matching what
# report.py suggests when it finds an unconfirmed recommendation.
if [[ -z "$CONFIRM_SEEDS" ]]; then
  for seed in $SEEDS; do CONFIRM_SEEDS="${CONFIRM_SEEDS}$((seed + 10)) "; done
fi
CONFIRM_SEEDS="$(echo "$CONFIRM_SEEDS" | xargs)"

# A confirmation seed that was also a selection seed confirms nothing -- it just
# re-reads a number already folded into the mean being checked.
for seed in $SEEDS; do
  for other in $CONFIRM_SEEDS; do
    if [[ "$seed" == "$other" ]]; then
      echo "Seed ${seed} is in both --seeds and --confirm-seeds." >&2
      exit 2
    fi
  done
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------ checkpoints
# The stats file travels with the BC *run*, not the individual checkpoint: it
# records the agent contract (frame_stack / frame_stride / action_chunk_size /
# latent_dim), which is identical for every epoch of one run. One stats file is
# therefore correct for all checkpoints in the sweep, and train_lewm reads the
# contract off it via _stats_contract().
if [[ -n "$BC_DIR" && -z "$BC_CHECKPOINTS" ]]; then
  # Exclude the *_stats.pth sidecar; sorted so epochs sweep in order. LC_ALL=C
  # keeps this byte-ordered, matching Python's sorted() in evaluate_sweep.py --
  # a locale-collated sort ignores punctuation and orders the unsuffixed final
  # checkpoint differently from the discovery side.
  BC_CHECKPOINTS="$(ls "$BC_DIR"/*.pth 2>/dev/null | grep -v '_stats\.pth$' | LC_ALL=C sort || true)"
  if [[ -z "$BC_STATS" ]]; then
    BC_STATS="$(ls "$BC_DIR"/*_stats.pth 2>/dev/null | head -1 || true)"
  fi
fi

if [[ -z "$BC_CHECKPOINTS" ]]; then
  echo "No BC checkpoints. Pass --bc-dir <dir> or --bc-checkpoints \"a.pth b.pth\"." >&2
  echo "Produce them with: python src/bc/train_bc_latent.py --epochs 200 --save_interval 20 ..." >&2
  exit 2
fi
if [[ -z "$BC_STATS" ]]; then
  echo "No BC stats file. Pass --bc-stats <path>." >&2
  exit 2
fi

# Sanity: fewer than ~5 checkpoints cannot distinguish "monotone in BC training"
# from "peaked in the middle", which is the entire shape you are trying to read.
NUM_CKPT="$(echo "$BC_CHECKPOINTS" | wc -w | tr -d ' ')"
if [[ "$NUM_CKPT" -lt 5 ]]; then
  echo "WARNING: only ${NUM_CKPT} BC checkpoints. 5-6 spanning underfit..overfit is" >&2
  echo "         the minimum that shows the shape of the curve rather than an argmax." >&2
fi

# --------------------------------------------------------------------- reward
# Held FIXED across the sweep. See note 2 in the header before changing this.
REWARD_ARGS=(
  --reward_mode dense
  --dense_reward_checkpoint models/probes/pusht_dense_reward_hf/dense_reward_classifier.pt
  --dense_reward_mode potential
  --dense_reward_coef 0.05
  --dense_reward_weights "1 0.75 0.4 0.1"
  --dense_reward_clip 0.5
)

# Tag derivation must match _bc_tag() in scripts/bc_selection/evaluate_sweep.py:
# the checkpoint's filename stem, with anything non-alphanumeric folded to "_".
bc_tag() {
  local stem
  stem="$(basename "$1")"
  stem="${stem%.pth}"
  echo "$stem" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//'
}

run_one() {
  local bc_ckpt="$1" seed="$2"
  local tag exp_name
  tag="$(bc_tag "$bc_ckpt")"
  exp_name="${EXP_PREFIX}_${tag}"

  echo "=== dream PPO | bc=${bc_ckpt} | seed ${seed} | exp=${exp_name} ==="
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi

  conda run --no-capture-output -n "$CONDA_ENV" \
  python -m src.ppo.train_lewm \
    --exp_name "$exp_name" \
    --seed "$seed" \
    --bc_checkpoint "$bc_ckpt" \
    --bc_stats "$BC_STATS" \
    --hidden_dim 256 \
    --frame_stack 3 \
    --frame_stride 5 \
    --action_chunk_size 5 \
    --wm_frameskip 5 \
    --fixed_target \
    --block_start_near_goal \
    --block_start_radius 200.0 \
    ${REWARD_ARGS[@]+"${REWARD_ARGS[@]}"} \
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
    --snapshot-interval 10
}

FAILED=()

# Checkpoint-major order: every seed of one BC checkpoint finishes together, so
# an interrupted sweep leaves whole cells rather than a ragged grid that the
# report would average unevenly.
train_grid() {
  local checkpoints="$1" seeds="$2"
  local bc_ckpt seed
  for bc_ckpt in $checkpoints; do
    for seed in $seeds; do
      # Deliberately not fatal -- see the note on failures in the header.
      if ! run_one "$bc_ckpt" "$seed"; then
        echo "!!! FAILED: bc=$(basename "$bc_ckpt") seed=${seed} -- continuing" >&2
        FAILED+=("$(basename "$bc_ckpt"):${seed}")
      fi
    done
  done
}

# Evaluate one phase and re-print the whole table. Assembled as an argv that is
# either printed or executed, so what --dry-run shows is by construction what
# would have run.
evaluate_and_report() {
  local role="$1" checkpoints="$2" seeds="$3"
  local eval_args report_args
  eval_args=(--bc-stats "$BC_STATS" --exp-prefix "$EXP_PREFIX" --output-root "$OUTPUT_ROOT")
  # Selection re-discovers the directory so every checkpoint gets a standalone
  # BC row; confirmation names the one checkpoint it re-trained.
  if [[ "$role" == "select" && -n "$BC_DIR" ]]; then
    eval_args+=(--bc-dir "$BC_DIR")
  else
    eval_args+=(--bc-checkpoints $checkpoints)
  fi
  # Only the seeds this phase actually trained, in their role. Confirmation
  # sends an empty --select-seeds so it does not re-walk the selection grid,
  # whose rows are already in results.jsonl.
  if [[ "$role" == "confirm" ]]; then
    eval_args+=(--select-seeds --confirm-seeds $seeds)
  else
    eval_args+=(--select-seeds $seeds)
  fi

  report_args=(--results "${OUTPUT_ROOT}/results.jsonl" --output-root "$OUTPUT_ROOT")
  if [[ "$WANDB" == "1" ]]; then report_args+=(--wandb); fi

  if [[ "$DRY_RUN" == "1" || "$RUN_EVAL" == "0" ]]; then
    echo "Would evaluate and report (${role}):"
    echo "  python -m scripts.bc_selection.evaluate_sweep ${eval_args[*]}"
    echo "  python -m scripts.bc_selection.report ${report_args[*]}"
    return 0
  fi

  echo "=== evaluating (${role}) ==="
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m scripts.bc_selection.evaluate_sweep "${eval_args[@]}"
  echo
  echo "=== report (${role}) ==="
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m scripts.bc_selection.report "${report_args[@]}"
}

# The winning checkpoint, read back from the manifest report.py just wrote --
# the same value the printed table flags as RECOMMENDED.
recommended_checkpoint() {
  conda run -n "$CONDA_ENV" python -c '
import json, sys
from pathlib import Path

manifest = Path(sys.argv[1]) / "selection_manifest.json"
if manifest.is_file():
    print(json.load(manifest.open())["recommended"].get("bc_checkpoint") or "")
' "$OUTPUT_ROOT" 2>/dev/null | tail -1
}

# ------------------------------------------------------------------- selection
train_grid "$BC_CHECKPOINTS" "$SEEDS"

echo
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "${#FAILED[@]} run(s) failed: ${FAILED[*]}" >&2
  echo "Only the cells that completed are represented below; re-run the failed" >&2
  echo "(checkpoint, seed) pairs before trusting the recommendation, since an" >&2
  echo "unevenly-seeded grid weights checkpoints differently." >&2
  echo >&2
fi

evaluate_and_report "$SEED_ROLE" "$BC_CHECKPOINTS" "$SEEDS"

# ---------------------------------------------------------------- confirmation
# Chained automatically: the winner is picked by the plateau rule, not by a
# human, so there is nothing to stop and decide. Skipped when the grid is
# incomplete -- confirming a winner drawn from partial data measures nothing.
if [[ "$SEED_ROLE" == "select" && "$AUTO_CONFIRM" == "1" \
      && "$RUN_EVAL" == "1" && "$DRY_RUN" != "1" ]]; then
  if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo >&2
    echo "Skipping the confirmation pass: ${#FAILED[@]} selection run(s) failed, so the" >&2
    echo "recommendation above comes from an incomplete grid. Re-run those cells, then:" >&2
    echo "  bash $0 --bc-dir '${BC_DIR}' --seeds '${SEEDS}'" >&2
  else
    WINNER="$(recommended_checkpoint)"
    if [[ -z "$WINNER" || ! -f "$WINNER" ]]; then
      echo >&2
      echo "Skipping the confirmation pass: no usable recommendation in" >&2
      echo "${OUTPUT_ROOT}/selection_manifest.json." >&2
    else
      echo
      echo "########################################################################"
      echo "# CONFIRMATION PASS: $(basename "$WINNER") on seeds ${CONFIRM_SEEDS}"
      echo "# Fresh seeds, because the selection mean above is the max over"
      echo "# checkpoints of a noisy quantity and is therefore biased upward."
      echo "########################################################################"
      echo
      train_grid "$WINNER" "$CONFIRM_SEEDS"
      evaluate_and_report "confirm" "$WINNER" "$CONFIRM_SEEDS"
    fi
  fi
fi

echo
echo "To publish the selected checkpoint to the Hub:"
echo "  python -m scripts.bc_selection.report --output-root ${OUTPUT_ROOT}$(
  [[ "$WANDB" == "1" ]] && printf ' --wandb'
) --push-to-hf"

# Non-zero only at the very end, so a partial grid still produces its report
# before the failure is surfaced to whatever launched this.
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  exit 1
fi
