#!/usr/bin/env bash
#SBATCH -J node2_full_upgrade
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 6:00:00
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
RUN_NAME="${RUN_NAME:-anuga_node2_full_upgrade_${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}"

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
  echo "[$(date -u +%F' '%T)] log_file=${LOG_FILE}"
} | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -c "import sys; import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node2.yaml \
  --run-name "${RUN_NAME}" 2>&1 | tee -a "${LOG_FILE}"
