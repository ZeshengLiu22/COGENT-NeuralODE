#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_NAME=no_relative_time
export ABLATION_CONFIG=configs/NODE2_Upgrade1_Ablation/model_node2_no_relative_time.yaml
exec bash "${SCRIPT_DIR}/_run_issm_node2_upgrade1_ablation_local.sh"
