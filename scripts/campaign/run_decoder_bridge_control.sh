#!/usr/bin/env bash
#
# Decoder-bridge dream PPO: the seeds the bridge comparison is missing.
#
# Dream PPO has been run with both bridges, but the decoder side is n=1 per prior
# while the de-projector side has 2-3 seeds. Seed 1 already exists for both priors
# (pushed under <namespace>/dream_sparse_decoder/seed1), so this fills seeds 2 and 3:
#
#   --prior legacy   from bc/pusht-bc-raw-cls-256-legacy      -> rawcls256_legacy/
#   --prior new      from bc/pusht-bc-raw-cls (best)          -> rawcls_bc_best/
#
# Split across two machines by giving each one --prior. Everything else is
# identical to the de-projector arms in run_rawcls_grid.sh, so the only variable
# against <namespace>/dream_sparse/seed<N> is --bridge.
#
# What this settles: on the new prior the decoder scored 0.553 against the
# de-projector's 0.707 mean, while on the legacy prior the two were within noise
# (0.747 vs 0.727). That is either a prior x bridge interaction -- the decoder
# being brittle to the prior, the de-projector robust -- or one unlucky run. With
# n=1 in the surprising cell there is no way to tell, and the bridge conclusion
# rests on it.
#
# Prerequisites (dream arms need more than the real ones):
#   * LeWM object checkpoint:   python -m scripts.download_lewm_checkpoint
#   * expert h5 at le-wm/models/datasets/pusht_expert_train.h5  (~46 GB)
#   * objective_met classifier under models/probes/pusht_lewm/
#     (hf.co/offline-rl-with-le-wm/probes/is_objective_met_probe_baseline.pt)
#   * .env with WANDB_API_KEY and HF_TOKEN
# The image decoder is pulled from the Hub automatically. All of the above is
# preflighted before the first run starts.
#
# Usage:
#   bash scripts/campaign/run_decoder_bridge_control.sh --prior legacy   # machine A, ~1 h
#   bash scripts/campaign/run_decoder_bridge_control.sh --prior new      # machine B, ~1 h
#   bash scripts/campaign/run_decoder_bridge_control.sh                  # both, sequential
#   bash scripts/campaign/run_decoder_bridge_control.sh --prior new --seeds "2" --dry-run
#
# Any other run_rawcls_grid.sh flag is passed through (--seeds, --no-push, --dry-run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PRIOR="both"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prior) PRIOR="$2"; shift 2 ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

case "$PRIOR" in
  legacy|new|both) ;;
  *) echo "--prior must be legacy, new or both (got $PRIOR)" >&2; exit 2 ;;
esac

LEGACY_DIR="hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls-256-legacy"
NEW_DIR="hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls"
DECODER="hf://offline-rl-with-le-wm/decoder_lewm_pusht/decoder_lewm_pusht.pt"

# Seeds 2 and 3: seed 1 is already published for both priors.
HAS_SEEDS=0
for arg in ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; do
  [[ "$arg" == "--seeds" ]] && HAS_SEEDS=1
done
SEED_ARGS=()
if [[ "$HAS_SEEDS" -eq 0 ]]; then
  SEED_ARGS=(--seeds "2 3")
fi

run_prior() {
  local label="$1" bc_dir="$2" bc_stem="$3" namespace="$4" exp_prefix="$5"
  echo ""
  echo "##############################################################"
  echo "## decoder bridge | ${label} prior | -> ${namespace}/dream_sparse_decoder"
  echo "##############################################################"
  bash scripts/campaign/run_rawcls_grid.sh \
    --only dream_sparse \
    --bridge decoder \
    --arm-suffix _decoder \
    --exp-prefix "$exp_prefix" \
    --hf-namespace "$namespace" \
    --bc-checkpoint "${bc_dir}/${bc_stem}.pth" \
    --bc-stats "${bc_dir}/${bc_stem}_stats.pth" \
    --decoder-checkpoint "$DECODER" \
    ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
}

if [[ "$PRIOR" == "legacy" || "$PRIOR" == "both" ]]; then
  run_prior legacy "$LEGACY_DIR" pusht_latent_bc rawcls256_legacy rawcls256_legacy
fi
if [[ "$PRIOR" == "new" || "$PRIOR" == "both" ]]; then
  run_prior new "$NEW_DIR" pusht_bc_raw_cls_best rawcls_bc_best rawcls
fi
