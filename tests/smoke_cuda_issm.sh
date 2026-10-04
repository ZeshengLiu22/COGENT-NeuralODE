#!/usr/bin/env bash
#SBATCH -J cogent_smoke_issm
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 12:00:00
#SBATCH --chdir=/home1/09575/zeshengliu/scratch/COGENT-NeuralODE
#SBATCH -o /home1/09575/zeshengliu/scratch/COGENT-NeuralODE/logs/cuda_smoke_issm_%j.out
#SBATCH -e /home1/09575/zeshengliu/scratch/COGENT-NeuralODE/logs/cuda_smoke_issm_%j.err
# Diagnostic only: canonical ISSM H1/K180/S60, full architecture, TC0.
# One bounded epoch: four training batches per rank; all held-out rollouts.
set -euo pipefail
PROJECT_ROOT="/home1/09575/zeshengliu/scratch/COGENT-NeuralODE"
cd "$PROJECT_ROOT"
PYTHON_BIN="/work/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
OUTPUT_DIR="$PROJECT_ROOT/outputs/cleanup_cuda_smoke_issm_${SLURM_JOB_ID}"
echo "Canonical ISSM CUDA smoke, four GPU processes, output_dir=$OUTPUT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/check_real_cuda_smoke.py --dataset issm --train-batches 4 --output-dir "$OUTPUT_DIR"
