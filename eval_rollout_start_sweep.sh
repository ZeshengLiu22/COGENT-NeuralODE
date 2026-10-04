#!/usr/bin/env bash
set -euo pipefail

# Each start reanchors the same checkpoint on true history. Slicing the resulting
# complete predictions later is handled by scripts/postprocess_rollout.py.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
  if [[ "$argument" == --known-steps ]]; then
    echo "This sweep fixes rollout starts at 60, 90, and 120." >&2
    exit 1
  fi
done
for known in 60 90 120; do
  bash "$PROJECT_ROOT/eval_rollout_tmux.sh" --inside-tmux "$@" \
    --config "configs/ablations/rollout_start/known${known}.yaml" --known-steps "$known"
done
