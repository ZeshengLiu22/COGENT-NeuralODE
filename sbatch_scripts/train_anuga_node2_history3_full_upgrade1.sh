#!/usr/bin/env bash
#SBATCH -J anuga_node2_h3_up1
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 24:00:00
#SBATCH -o /home1/09575/zeshengliu/scratch/continuous_graph_emulator/logs/slurm_%x_%j.out
#SBATCH -e /home1/09575/zeshengliu/scratch/continuous_graph_emulator/logs/slurm_%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=zel220@lehigh.edu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home1/09575/zeshengliu/scratch/continuous_graph_emulator}"
SCRIPT_DIR="${PROJECT_ROOT}/sbatch_scripts"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_activate_torchpyg_cge.sh"

cd "${PROJECT_ROOT}"

mkdir -p logs outputs

NPROC="${NPROC:-${SLURM_NTASKS_PER_NODE:-4}}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-anuga_node2_history3_full_upgrade1_${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}"

BASE_CONFIG="${BASE_CONFIG:-configs/ANUGA_History_Scan/base_ANUGA_history3.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/anuga.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_node2.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-}"

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

{
  echo "[$(date -u +%F' '%T)] project_root=${PROJECT_ROOT}"
  echo "[$(date -u +%F' '%T)] run_name=${RUN_NAME}"
  echo "[$(date -u +%F' '%T)] python_bin=${PYTHON_BIN}"
  echo "[$(date -u +%F' '%T)] nproc=${NPROC}"
  echo "[$(date -u +%F' '%T)] base_config=${BASE_CONFIG}"
  echo "[$(date -u +%F' '%T)] dataset_config=${DATASET_CONFIG}"
  echo "[$(date -u +%F' '%T)] model_config=${MODEL_CONFIG}"
  echo "[$(date -u +%F' '%T)] extra_configs=${EXTRA_CONFIGS:-<none>}"
  echo "[$(date -u +%F' '%T)] log_file=${LOG_FILE}"
} | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -c "import sys; import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "${LOG_FILE}"

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

"${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
