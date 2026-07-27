#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

all_ablations=(
  upgrade_off
  residual_only
  history_in_ode_only
  relative_time_only
  no_residual
  no_history_in_ode
  no_relative_time
  full_upgrade
)

if [[ "$#" -gt 0 ]]; then
  ablations=("$@")
else
  ablations=("${all_ablations[@]}")
fi

for ablation in "${ablations[@]}"; do
  script="${SCRIPT_DIR}/train_issm_node2_ablate_${ablation}.sh"
  if [[ ! -f "${script}" ]]; then
    echo "Unknown ablation or missing script: ${ablation}" >&2
    exit 1
  fi
  echo "Submitting ${script} with HISTORY_LEN=${HISTORY_LEN:-4}"
  sbatch "${script}"
  sleep "${SUBMIT_SLEEP_SECONDS:-1}"
done
