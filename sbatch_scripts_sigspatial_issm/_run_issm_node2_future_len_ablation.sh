#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${FUTURE_LEN:-}" =~ ^[0-9]+$ ]]; then
  echo "FUTURE_LEN must be set to a positive integer, got: ${FUTURE_LEN:-<unset>}" >&2
  exit 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${THIS_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/work/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python}"

cd "${PROJECT_ROOT}"

mkdir -p logs outputs

HISTORY_LEN="${HISTORY_LEN:-6}"
NPROC="${NPROC:-${SLURM_NTASKS_PER_NODE:-4}}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_VARIANT="${RUN_VARIANT:-u11}"
RUN_NAME="${RUN_NAME:-sigspatial_issm_node2_h${HISTORY_LEN}_future${FUTURE_LEN}_${RUN_VARIANT}_${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}"

FUTURE_CONFIG_ROOT="${FUTURE_CONFIG_ROOT:-${PROJECT_ROOT}/configs/ISSM_Future_Len_Ablation}"
BASE_CONFIG="${BASE_CONFIG:-${FUTURE_CONFIG_ROOT}/base_ISSM_history${HISTORY_LEN}_future${FUTURE_LEN}.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/issm.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_node2.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-configs/ISSM_History_Scan/issm_pig5000_fast_loader.yaml}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
else
  MASTER_PORT="${MASTER_PORT:-29500}"
fi
export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

{
  echo "[$(date -u +%F' '%T)] project_root=${PROJECT_ROOT}"
  echo "[$(date -u +%F' '%T)] slurm_job_id=${SLURM_JOB_ID:-<none>}"
  echo "[$(date -u +%F' '%T)] history_len=${HISTORY_LEN}"
  echo "[$(date -u +%F' '%T)] future_len=${FUTURE_LEN}"
  echo "[$(date -u +%F' '%T)] run_name=${RUN_NAME}"
  echo "[$(date -u +%F' '%T)] python_bin=${PYTHON_BIN}"
  echo "[$(date -u +%F' '%T)] nproc=${NPROC}"
  echo "[$(date -u +%F' '%T)] master_addr=${MASTER_ADDR}"
  echo "[$(date -u +%F' '%T)] master_port=${MASTER_PORT}"
  echo "[$(date -u +%F' '%T)] base_config=${BASE_CONFIG}"
  echo "[$(date -u +%F' '%T)] dataset_config=${DATASET_CONFIG}"
  echo "[$(date -u +%F' '%T)] model_config=${MODEL_CONFIG}"
  echo "[$(date -u +%F' '%T)] extra_configs=${EXTRA_CONFIGS:-<none>}"
  echo "[$(date -u +%F' '%T)] train_args=${TRAIN_ARGS:-<none>}"
  echo "[$(date -u +%F' '%T)] log_file=${LOG_FILE}"
} | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -c "import sys; import torch; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "${LOG_FILE}"

train_cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  scripts/train.py
  --config "${BASE_CONFIG}"
  --config "${DATASET_CONFIG}"
  --config "${MODEL_CONFIG}"
)
for extra_config in ${EXTRA_CONFIGS}; do
  train_cmd+=(--config "${extra_config}")
done
train_cmd+=(--run-name "${RUN_NAME}")
for train_arg in ${TRAIN_ARGS}; do
  train_cmd+=("${train_arg}")
done

"${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
