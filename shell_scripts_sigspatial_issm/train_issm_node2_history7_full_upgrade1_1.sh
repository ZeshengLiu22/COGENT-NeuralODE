#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HISTORY_LEN=7
exec bash "${SCRIPT_DIR}/_run_issm_node2_history_scan_local.sh"
