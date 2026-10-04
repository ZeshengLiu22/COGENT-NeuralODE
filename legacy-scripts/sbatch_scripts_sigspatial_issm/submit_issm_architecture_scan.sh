#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/sweep_config.sh"
require_history
export HISTORY_LEN

for ENCODER in transformer lstm; do
  for RESIDUAL in on off; do
    for ODE_CONTEXT in on off; do
      for RELATIVE_TIME in on off; do
        export ENCODER RESIDUAL ODE_CONTEXT RELATIVE_TIME
        configure_sweep issm architecture
        echo "Architecture H=${HISTORY_LEN}: ${ENCODER}, residual=${RESIDUAL}, context=${ODE_CONTEXT}, time=${RELATIVE_TIME}"
        sbatch --job-name="issm_arch_${SWEEP_LABEL}" "${SCRIPT_DIR}/train_issm_node2_architecture.sh"
        sleep "${SUBMIT_SLEEP_SECONDS:-1}"
      done
    done
  done
done
