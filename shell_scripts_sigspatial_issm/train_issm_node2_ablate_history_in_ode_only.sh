#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_NAME=history_in_ode_only
export ABLATION_CONFIG=configs/NODE2_Upgrade1_Ablation/model_node2_history_in_ode_only.yaml
exec bash "${SCRIPT_DIR}/_run_issm_node2_upgrade1_ablation_local.sh"
