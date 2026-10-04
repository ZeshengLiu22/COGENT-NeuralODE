#!/usr/bin/env bash
#SBATCH -J sig_future90
#SBATCH -p h100
##SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 9:00:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err
##SBATCH --mail-type=all
##SBATCH --mail-user=zel220@lehigh.edu

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FUTURE_LEN=90
exec bash "${SCRIPT_DIR}/_run_issm_node2_future_len_ablation.sh"
