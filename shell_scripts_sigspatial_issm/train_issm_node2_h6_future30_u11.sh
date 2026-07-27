#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HISTORY_LEN=6
export FUTURE_LEN=30
exec bash "${SCRIPT_DIR}/_run_issm_node2_future_len_ablation_local.sh"
