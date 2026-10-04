#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/sweep_config.sh"
require_history
export HISTORY_LEN
require_architecture
require_future
export ENCODER RESIDUAL ODE_CONTEXT RELATIVE_TIME FUTURE_LEN

for TC_MODE in tc0_off tc1_adjacent_increment tc2_random_pair_increment tc3_multiscale_rate tc4_rate_curvature tc5_hybrid; do
  export TC_MODE
  configure_sweep issm temporal_consistency
  echo "Temporal consistency: ${TC_MODE} with ${SWEEP_LABEL}"
  sbatch --job-name="issm_${SWEEP_LABEL}" "${SCRIPT_DIR}/train_issm_node2_temporal_consistency.sh"
  sleep "${SUBMIT_SLEEP_SECONDS:-1}"
done
