#!/usr/bin/env bash
#SBATCH -J cercat_anuga_h1_u11
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 06:00:00
#SBATCH --chdir=/scratch/09575/zeshengliu/COGENT-NeuralODE
#SBATCH -o /scratch/09575/zeshengliu/COGENT-NeuralODE/logs/slurm_%x_%j.out
#SBATCH -e /scratch/09575/zeshengliu/COGENT-NeuralODE/logs/slurm_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zel220@lehigh.edu

set -euo pipefail
PROJECT_ROOT="/scratch/09575/zeshengliu/COGENT-NeuralODE"
RUNNER="${PROJECT_ROOT}/sbatch_scripts_cercat_anuga/_run_anuga_node2_history_scan.sh"
export PROJECT_ROOT
export HISTORY_LEN=1
exec bash "${RUNNER}"
