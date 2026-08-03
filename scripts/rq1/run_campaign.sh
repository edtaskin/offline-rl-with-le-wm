#!/usr/bin/env bash
# RQ1: can policy improvement happen entirely inside the world model?
#
# Trains the two PPO variants across matched seeds, evaluates everything on one
# protocol, and writes the table plus figures.
#
# The two arms differ in *two* things, not one, and the second is not free:
#   1. where the rollouts come from -- real simulator vs imagined LeWM rollouts;
#   2. where the shaping reward comes from -- the real arm reads the env's native
#      dense reward, the dream arm has no env to ask, so it scores imagined
#      latents with the learned classifier from scripts/probes and applies it as
#      potential shaping on top of sparse success.
# Everything else below is deliberately identical between the two blocks. Read
# any dream-vs-real gap as the joint effect of (1) and (2).
#
#   bash scripts/rq1/run_campaign.sh              # full campaign
#   SEEDS="1" bash scripts/rq1/run_campaign.sh    # one seed, for a dry run
#   STAGES="evaluate report" bash scripts/rq1/run_campaign.sh   # skip training
#
# Training is not resumable (the trainer has no restart path), but evaluation is:
# the ledger is append-only and already-evaluated checkpoints are skipped, so
# re-running after an interruption picks up where it stopped.
set -euo pipefail

cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-1 2 3}"
STAGES="${STAGES:-train_real train_dream evaluate report}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
LOG_DIR="${LOG_DIR:-runs/rq1/logs}"

# Experiment names; the run dirs they produce (runs/<exp>__seed<k>/<stamp>) are
# what the evaluate stage discovers.
EXP_REAL="${EXP_REAL:-rq1_real}"
EXP_DREAM="${EXP_DREAM:-rq1_dream}"

# Cadence of checkpointing and of the selection evaluations. Kept equal so every
# dream snapshot has a same-iteration imagined score to pair with.
INTERVAL="${INTERVAL:-20}"
# Permanent step-tagged snapshots. The dream arm needs them for the selection
# scatter; the real arm ships with them OFF (0), matching the run that produced
# the current checkpoints. Set REAL_SNAPSHOT_INTERVAL=${INTERVAL} if you want the
# interaction-budget curve -- without snapshots the curve has no real-PPO points.
REAL_SNAPSHOT_INTERVAL="${REAL_SNAPSHOT_INTERVAL:-0}"
DREAM_SNAPSHOT_INTERVAL="${DREAM_SNAPSHOT_INTERVAL:-${INTERVAL}}"
# Evaluation protocol; shrink both for a cheap end-to-end rehearsal of the
# pipeline, e.g. SEEDS=1 TOTAL_TIMESTEPS=15000 INTERVAL=1 EPISODES=3 REPEATS=1.
EPISODES="${EPISODES:-50}"
REPEATS="${REPEATS:-3}"

BC_CHECKPOINT="${BC_CHECKPOINT:-hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth}"
BC_STATS="${BC_STATS:-hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth}"
# Learned dense-reward classifier for the dream arm (local copy of the HF probe).
DENSE_REWARD_CHECKPOINT="${DENSE_REWARD_CHECKPOINT:-models/probes/pusht_dense_reward_hf/dense_reward_classifier.pt}"

# Hugging Face upload of each run's best.pt. Each seed gets its own subdirectory:
# a bare prefix would make the seeds overwrite each other in the repo.
PUSH_TO_HF="${PUSH_TO_HF:-1}"
HF_REPO_ID="${HF_REPO_ID:-offline-rl-with-le-wm/ppo}"
HF_PREFIX_REAL="${HF_PREFIX_REAL:-real_env_dense}"
HF_PREFIX_DREAM="${HF_PREFIX_DREAM:-world_model_sparse_dense_correlation}"

hf_args() {  # hf_args <prefix> <seed>
  [[ "${PUSH_TO_HF}" == "1" ]] || return 0
  printf '%s\n' --push_to_hf --hf_repo_id "${HF_REPO_ID}" --hf_path_prefix "$1/seed$2"
}

# Shared task definition and agent contract. block_start_radius fixes the start
# distribution that both trainers and the evaluator use; without it the three
# agents would be scored on a task none of them trained on. bc_penalty is passed
# explicitly because its dataclass default has been flipped mid-project -- both
# published runs trained with it off.
TASK_ARGS=(
  --fixed_target
  --block_start_near_goal
  --block_start_radius 200.0
  --reward_mode dense
  --no-bc-penalty
  --gamma 0.99
  --hidden_dim 256
  --frame_stack 3
  --frame_stride 5
  --action_chunk_size 5
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --num_envs 8
  --num_chunks 64
  --bc_checkpoint "${BC_CHECKPOINT}"
  --bc_stats "${BC_STATS}"
  --log_interval 10
  --save_interval "${INTERVAL}"
)

# Evaluation protocol. The headline table gets the full 3x50 episodes; the
# budget curve trades episodes for points, since a curve's shape survives more
# noise per point than a headline number does.
HEADLINE_EVAL=(--episodes "${EPISODES}" --repeats "${REPEATS}" --seed 42 --block-start-radius 200)
CURVE_EVAL=(--episodes "${EPISODES}" --repeats 1 --seed 42 --block-start-radius 200)

mkdir -p "${LOG_DIR}" runs/rq1
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# ---------------------------------------------------------------- training
if has_stage train_real; then
  for seed in ${SEEDS}; do
    echo "=== real-env PPO | seed ${seed} ==="
    # reward_mode dense here is the *env's* native dense reward -- no classifier
    # is involved. eval_interval selects best.pt from held-out real episodes: the
    # standard, interaction-paying baseline. Those steps are charged to its budget. 
    python src/ppo/train.py \
      --exp_name "${EXP_REAL}" --seed "${seed}" \
      "${TASK_ARGS[@]}" \
      --snapshot_interval "${REAL_SNAPSHOT_INTERVAL}" \
      --eval_interval 20 --eval_episodes 20 --eval_seed 0 \
      $(hf_args "${HF_PREFIX_REAL}" "${seed}") \
      2>&1 | tee "${LOG_DIR}/train_real_seed${seed}.log"
  done
fi

if has_stage train_dream; then
  for seed in ${SEEDS}; do
    echo "=== dream PPO | seed ${seed} ==="
    # selection dream is the point of the experiment: best.pt is ranked by
    # imagined success on held-out expert anchors, costing zero env steps.
    # record_real_eval logs the real held-out metric alongside it for the
    # correlation figure -- diagnostic only, but its env steps ARE charged, so
    # this run's interaction budget is not zero. It also implicitly sets
    # eval_interval to dream_eval_interval.
    # dream_snapshot_interval matches dream_eval_interval so every saved
    # checkpoint has an imagined score to pair with later.
    python -m src.ppo.train_lewm \
      --exp_name "${EXP_DREAM}" --seed "${seed}" \
      "${TASK_ARGS[@]}" \
      --wm_frameskip 5 \
      --snapshot_interval "${DREAM_SNAPSHOT_INTERVAL}" \
      --dense_reward_checkpoint "${DENSE_REWARD_CHECKPOINT}" \
      --dense_reward_mode potential \
      --dense_reward_coef 0.05 \
      --dense_reward_weights "1 0.75 0.4 0.1" \
      --dense_reward_clip 0.5 \
      --selection dream \
      --dream_eval_interval 20 --dream_eval_episodes 96 --dream_eval_seed 12345 \
      --dream_val_fraction 0.1 --dream_val_split_seed 0 \
      --record-real-eval \
      --eval_episodes 20 --eval_seed 0 \
      $(hf_args "${HF_PREFIX_DREAM}" "${seed}") \
      2>&1 | tee "${LOG_DIR}/train_dream_seed${seed}.log"
  done
fi

# -------------------------------------------------------------- evaluation
latest_run_dir() {  # newest timestamped dir under runs/<exp>__seed<k>/
  local dir="runs/$1__seed$2"
  [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1
}

if has_stage evaluate; then
  real_runs=() dream_runs=()
  for seed in ${SEEDS}; do
    real_dir="$(latest_run_dir "${EXP_REAL}" "${seed}" || true)"
    dream_dir="$(latest_run_dir "${EXP_DREAM}" "${seed}" || true)"
    [[ -n "${real_dir}" ]] && real_runs+=(--run "real-ppo=${real_dir}")
    [[ -n "${dream_dir}" ]] && dream_runs+=(--run "dream-ppo=${dream_dir}")
  done

  echo "=== headline evaluation (best + final) ==="
  python -m scripts.rq1.evaluate_grid \
    --bc "${BC_CHECKPOINT}" --bc-stats "${BC_STATS}" \
    ${real_runs[@]+"${real_runs[@]}"} ${dream_runs[@]+"${dream_runs[@]}"} \
    --checkpoints best final \
    "${HEADLINE_EVAL[@]}"

  echo "=== interaction-budget curve + selection pairs (snapshots) ==="
  python -m scripts.rq1.evaluate_grid \
    ${real_runs[@]+"${real_runs[@]}"} ${dream_runs[@]+"${dream_runs[@]}"} \
    --checkpoints snapshots \
    "${CURVE_EVAL[@]}"
fi

# ------------------------------------------------------------------ report
if has_stage report; then
  echo "=== report ==="
  python -m scripts.rq1.report --headline-checkpoint best
  python -m scripts.rq1.report --headline-checkpoint final --output-dir runs/rq1/final_selection
fi

echo "Done. Table and figures under runs/rq1/."
