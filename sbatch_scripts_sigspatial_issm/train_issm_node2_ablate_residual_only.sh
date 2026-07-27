#!/usr/bin/env bash
#SBATCH -J sig_ab_res
#SBATCH -p h100
##SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 9:00:00
#SBATCH -o /home/exouser/Documents/NODE-ISSM/continuous_graph_emulator_issm/logs/slurm_%x_%j.out
#SBATCH -e /home/exouser/Documents/NODE-ISSM/continuous_graph_emulator_issm/logs/slurm_%x_%j.err
##SBATCH --mail-type=all
##SBATCH --mail-user=zel220@lehigh.edu
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_NAME=residual_only
export ABLATION_CONFIG=configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml
exec bash "${SCRIPT_DIR}/_run_issm_node2_upgrade1_ablation.sh"
