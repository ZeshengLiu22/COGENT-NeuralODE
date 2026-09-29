#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${HISTORY_LEN:-}" =~ ^[1-8]$ ]]; then
  echo "HISTORY_LEN must be set to an integer from 1 to 8, got: ${HISTORY_LEN:-<unset>}" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/09575/zeshengliu/COGENT-NeuralODE}"
ACTIVATE_SCRIPT="${PROJECT_ROOT}/sbatch_scripts/_activate_torchpyg_cge.sh"
STAMPEDE3_PYTHON="/work2/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Activation script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN:-}" && -x "${STAMPEDE3_PYTHON}" ]]; then
  export PYTHON_BIN="${STAMPEDE3_PYTHON}"
fi

# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

NPROC="${NPROC:-${SLURM_NTASKS_PER_NODE:-4}}"
if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer, got: ${NPROC}" >&2
  exit 1
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_VARIANT="${RUN_VARIANT:-full_upgrade1_1}"
RUN_NAME="${RUN_NAME:-cercat_anuga_node2_history${HISTORY_LEN}_${RUN_VARIANT}_${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}"

BASE_CONFIG="${BASE_CONFIG:-configs/ANUGA_History_Scan/base_ANUGA_history${HISTORY_LEN}.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/anuga.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_node2.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-configs/ANUGA_History_Scan/anuga_fast_loader.yaml}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

for config_path in "${BASE_CONFIG}" "${DATASET_CONFIG}" "${MODEL_CONFIG}"; do
  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    exit 1
  fi
done
for extra_config in ${EXTRA_CONFIGS}; do
  if [[ ! -f "${extra_config}" ]]; then
    echo "Extra config not found: ${extra_config}" >&2
    exit 1
  fi
done

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
  echo "[$(date -u +%F' '%T)] host=$(hostname)"
  echo "[$(date -u +%F' '%T)] slurm_job_id=${SLURM_JOB_ID:-<none>}"
  echo "[$(date -u +%F' '%T)] slurm_job_nodelist=${SLURM_JOB_NODELIST:-<none>}"
  echo "[$(date -u +%F' '%T)] history_len=${HISTORY_LEN}"
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

"${PYTHON_BIN}" -c "import sys; import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())" 2>&1 | tee -a "${LOG_FILE}"

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
