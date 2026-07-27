#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SESSION_NAME="${SESSION_NAME:-anuga_node2}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/model_node2.yaml}"
export STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export RUN_NAME="${RUN_NAME:-anuga_node2_${STAMP}}"

exec bash "$PROJECT_ROOT/train_anuga.sh" "$@"
