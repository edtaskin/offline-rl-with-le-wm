#!/usr/bin/env bash
#
# Prior control: was the 0.92 real-env result the prior, or the seed?
#
# `hf://offline-rl-with-le-wm/ppo/agent_real_env.pt` scored 0.920 on the canonical
# protocol, trained from the legacy 256-wide raw-CLS BC. The rawcls campaign,
# identical in every hyperparameter but starting from
# bc/rawcls-bc/seed42/pusht_raw_cls_bc_best.pth, scores 0.833 and 0.720 at seeds
# 1 and 2. Two readings: the new prior is worse, or 0.920 was one lucky seed of a
# noisy pipeline. Nothing distinguishes them, because the old number is n=1 from a
# lineage whose bc_checkpoint reference has since been overwritten on the Hub.
#
# This runs the same real-env arms from the *legacy* prior, which the repo owner
# reconstructed and published, and which is bit-identical to the prior recovered
# from the old dream run's step-0 snapshot. Only the prior differs from the rawcls
# campaign, so the comparison isolates it:
#
#   legacy_real_sparse  seeds 1 2 3   vs   rawcls_real_sparse  seeds 1 2 3
#   legacy_real_dense   seeds 1 2 3   vs   rawcls_real_dense   seeds 1 2 3
#
# Only the real-env arms: the bridge plays no part here (real-env PPO encodes
# real frames straight to CLS), and 0.920 was a real-env number. No dream run has
# ever approached it.
#
# What this machine needs -- nothing large, and no expert h5:
#   * the LeWM object checkpoint:  python -m scripts.download_lewm_checkpoint
#   * .env with WANDB_API_KEY and HF_TOKEN (BC priors download automatically)
#   * requirements.txt installed
# The script preflights the encoder and every Hub destination before training.
#
# Usage:
#   bash scripts/campaign/run_legacy_prior_control.sh                 # both arms, seeds 1 2 3  (~5 h)
#   bash scripts/campaign/run_legacy_prior_control.sh --only real_sparse   # the 0.92 comparison only (~2.5 h)
#   bash scripts/campaign/run_legacy_prior_control.sh --dry-run
#
# Any flag of scripts/campaign/run_rawcls_grid.sh is accepted and passed through;
# --seeds "1 2 3" and --no-push are the useful ones.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LEGACY_DIR="hf://offline-rl-with-le-wm/bc/rawcls-bc/seed42"

# --only defaults to the two real arms rather than the grid's four: the dream arms
# would answer a different question and cost another 3 hours.
HAS_ONLY=0
for arg in "$@"; do
  [[ "$arg" == "--only" ]] && HAS_ONLY=1
done
ONLY_ARGS=()
if [[ "$HAS_ONLY" -eq 0 ]]; then
  ONLY_ARGS=(--only "real_sparse real_dense")
fi

exec bash scripts/campaign/run_rawcls_grid.sh \
  --exp-prefix legacy \
  --bc-checkpoint "${LEGACY_DIR}/pusht_raw_cls_bc_best.pth" \
  --bc-stats "${LEGACY_DIR}/pusht_raw_cls_bc_best_stats.pth" \
  ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"} \
  "$@"
