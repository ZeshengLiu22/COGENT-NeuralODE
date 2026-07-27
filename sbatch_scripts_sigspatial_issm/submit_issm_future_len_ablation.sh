#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$#" -gt 0 ]]; then
  future_lens=("$@")
else
  future_lens=(30 45 60 75 90 120 150 180)
fi

for future_len in "${future_lens[@]}"; do
  script="${SCRIPT_DIR}/train_issm_node2_h6_future${future_len}_u11.sh"
  if [[ ! -f "${script}" ]]; then
    echo "Unknown future length or missing script: ${future_len}" >&2
    exit 1
  fi
  echo "Submitting ${script}"
  sbatch "${script}"
  sleep "${SUBMIT_SLEEP_SECONDS:-1}"
done
