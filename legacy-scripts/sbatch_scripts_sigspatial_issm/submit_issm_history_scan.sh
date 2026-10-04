#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -gt 0 ]]; then
  histories=("$@")
else
  histories=(1 2 3 4 5 6 7 8)
fi

for history_len in "${histories[@]}"; do
  if [[ ! "${history_len}" =~ ^[1-8]$ ]]; then
    echo "History length must be an integer from 1 to 8, got: ${history_len}" >&2
    exit 1
  fi
  script="${SCRIPT_DIR}/train_issm_node2_history${history_len}_full_upgrade1_1.sh"
  echo "Submitting ${script}"
  sbatch "${script}"
  sleep "${SUBMIT_SLEEP_SECONDS:-1}"
done
