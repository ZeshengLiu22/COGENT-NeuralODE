#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/sweep_config.sh"
require_history
export HISTORY_LEN
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
require_architecture
require_future
export ENCODER RESIDUAL ODE_CONTEXT RELATIVE_TIME FUTURE_LEN
if [[ -z "${TMUX:-}" && "${CGE_AUTO_TMUX:-1}" != "0" && -t 0 && -t 1 ]]; then
  if command -v tmux >/dev/null 2>&1; then
    scans_for_name=("$@")
    if [[ "${#scans_for_name[@]}" -eq 0 ]]; then
      scans_for_name=(selected)
    fi
    TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-issm_experiment_$(IFS=_; echo "${scans_for_name[*]}")_$(date -u +%Y%m%d_%H%M%S)}"
    tmux_cmd=(env CGE_AUTO_TMUX=0 bash "${SCRIPT_PATH}" "$@")
    printf -v tmux_shell_cmd '%q ' "${tmux_cmd[@]}"
    tmux_shell_cmd+="; status=\$?; printf '\\n[experiment scan finished with exit code %s] Press Ctrl-d to close this tmux session.\\n' \"\$status\"; exec bash"
    echo "Starting tmux session: ${TMUX_SESSION_NAME}"
    exec tmux new-session -s "${TMUX_SESSION_NAME}" -c "${SCRIPT_DIR}/.." "${tmux_shell_cmd}"
  else
    echo "tmux not found; running in the current shell." >&2
  fi
fi

for TC_MODE in tc0_off tc1_adjacent_increment tc2_random_pair_increment tc3_multiscale_rate tc4_rate_curvature tc5_hybrid; do
  export TC_MODE
  configure_sweep issm temporal_consistency
  echo "Temporal consistency: ${TC_MODE} with ${SWEEP_LABEL}"
  bash "${SCRIPT_DIR}/_run_issm_node2_temporal_consistency_scan_local.sh"
done
