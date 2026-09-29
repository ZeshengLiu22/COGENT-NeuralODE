#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DEVICE="${DEVICE:-cuda}"
KNOWN_STEPS="${KNOWN_STEPS:-8}"
ANUGA_NUM_FRAMES="${ANUGA_NUM_FRAMES:-3}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

if [[ "$#" -gt 0 ]]; then
  histories=("$@")
else
  histories=(1 2 3 4 5 6 7 8)
fi

if [[ ! "${KNOWN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KNOWN_STEPS must be a positive integer, got: ${KNOWN_STEPS}" >&2
  exit 1
fi
if [[ ! "${ANUGA_NUM_FRAMES}" =~ ^[0-9]+$ ]]; then
  echo "ANUGA_NUM_FRAMES must be a nonnegative integer, got: ${ANUGA_NUM_FRAMES}" >&2
  exit 1
fi
if [[ "${SKIP_COMPLETED}" != "0" && "${SKIP_COMPLETED}" != "1" ]]; then
  echo "SKIP_COMPLETED must be 0 or 1, got: ${SKIP_COMPLETED}" >&2
  exit 1
fi

for history_len in "${histories[@]}"; do
  if [[ ! "${history_len}" =~ ^[1-8]$ ]]; then
    echo "History length must be an integer from 1 to 8, got: ${history_len}" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

for history_len in "${histories[@]}"; do
  shopt -s nullglob
  checkpoints=(
    "${PROJECT_ROOT}"/outputs/cercat_anuga_node2_history"${history_len}"_full_upgrade1_1_*/best.pt
  )
  shopt -u nullglob
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one history-${history_len} checkpoint, found ${#checkpoints[@]}." >&2
    printf '  %s\n' "${checkpoints[@]}" >&2
    exit 1
  fi

  checkpoint="${checkpoints[0]}"
  run_dir="$(dirname "${checkpoint}")"
  predictions_path="${run_dir}/best.full_rollout_predictions.npz"
  flood_summary_path="${run_dir}/best.full_rollout_flood_maps/summary.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -s "${predictions_path}" && -s "${flood_summary_path}" ]]; then
    echo "Skipping completed history-${history_len} rollout: ${run_dir}"
    continue
  fi

  log_file="${LOG_DIR}/$(basename "${run_dir}").full_rollout_k${KNOWN_STEPS}.log"
  echo "Starting history-${history_len} rollout: ${run_dir}"
  "${PYTHON_BIN}" scripts/run_full_rollout.py \
    --checkpoint "${checkpoint}" \
    --split test \
    --device "${DEVICE}" \
    --known-steps "${KNOWN_STEPS}" \
    --anuga-num-frames "${ANUGA_NUM_FRAMES}" \
    2>&1 | tee "${log_file}"

  if [[ ! -s "${predictions_path}" || ! -s "${flood_summary_path}" ]]; then
    echo "History-${history_len} evaluation exited without complete artifacts." >&2
    exit 1
  fi
done

echo "All requested ANUGA history rollouts are complete."
