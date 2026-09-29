#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/09575/zeshengliu/COGENT-NeuralODE"
SBATCH_DIR="${PROJECT_ROOT}/sbatch_scripts_cercat_anuga"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

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
done

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

for history_len in "${histories[@]}"; do
  script="${SBATCH_DIR}/train_anuga_node2_history${history_len}_full_upgrade1_1.sh"
  if [[ ! -f "${script}" ]]; then
    echo "SBATCH script not found: ${script}" >&2
    exit 1
  fi
  echo "Submitting ${script}"
  sbatch "${script}"
  sleep "${SUBMIT_SLEEP_SECONDS:-1}"
done
