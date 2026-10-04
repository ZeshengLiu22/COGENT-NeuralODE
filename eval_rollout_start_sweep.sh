#!/usr/bin/env bash
set -euo pipefail

# Three evaluation-only starts reanchor the same checkpoint on true history.
# Pass --checkpoint outputs/<final-run>/best.pt; no model is trained/reselected.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
# Reject a second rollout-start override so every output has its declared S.
case " $* " in
  *" --known-steps "*|*" --known-steps="*)
    echo "These robustness evaluations fix rollout starts at 60, 90, and 120." >&2
    exit 1 ;;
esac
bash "$PROJECT_ROOT/eval_rollout_tmux.sh" --inside-tmux "$@" \
  --config configs/ablations/issm/rollout_start/known60.yaml --known-steps 60
bash "$PROJECT_ROOT/eval_rollout_tmux.sh" --inside-tmux "$@" \
  --config configs/ablations/issm/rollout_start/known90.yaml --known-steps 90
bash "$PROJECT_ROOT/eval_rollout_tmux.sh" --inside-tmux "$@" \
  --config configs/ablations/issm/rollout_start/known120.yaml --known-steps 120
