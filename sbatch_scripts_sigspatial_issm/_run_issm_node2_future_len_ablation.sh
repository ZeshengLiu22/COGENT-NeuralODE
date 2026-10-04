#!/usr/bin/env bash
set -euo pipefail


THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${THIS_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/work/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python}"

source "${THIS_DIR}/../scripts/sweep_config.sh"
configure_sweep issm future

cd "${PROJECT_ROOT}"

mkdir -p logs outputs

NPROC="${NPROC:-${SLURM_NTASKS_PER_NODE:-4}}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-issm_node2_future_${SWEEP_LABEL}_${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}"

DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/issm.yaml}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/protocols/issm/main.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/node2.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG-configs/runtime/fast.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-}"
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
  echo "[$(date -u +%F' '%T)] run_name=${RUN_NAME}"
  echo "[$(date -u +%F' '%T)] python_bin=${PYTHON_BIN}"
  echo "[$(date -u +%F' '%T)] nproc=${NPROC}"
  echo "[$(date -u +%F' '%T)] dataset_config=${DATASET_CONFIG}"
  echo "[$(date -u +%F' '%T)] model_config=${MODEL_CONFIG}"
  echo "[$(date -u +%F' '%T)] runtime_config=${RUNTIME_CONFIG:-<none>}"
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
  --config configs/default.yaml
  --config "${DATASET_CONFIG}"
  --config "${PROTOCOL_CONFIG}"
  --config "${MODEL_CONFIG}"
)
for sweep_config in "${SWEEP_CONFIGS[@]}"; do
  train_cmd+=(--config "${sweep_config}")
done
for extra_config in ${EXTRA_CONFIGS}; do
  train_cmd+=(--config "${extra_config}")
done
if [[ -n "${RUNTIME_CONFIG}" ]]; then
  train_cmd+=(--config "${RUNTIME_CONFIG}")
fi
train_cmd+=(--run-name "${RUN_NAME}")
for train_arg in ${TRAIN_ARGS}; do
  train_cmd+=("${train_arg}")
done

"${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
