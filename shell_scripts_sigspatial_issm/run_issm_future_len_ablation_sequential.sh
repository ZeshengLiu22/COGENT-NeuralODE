#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/sweep_config.sh"
require_history
require_architecture
export HISTORY_LEN ENCODER RESIDUAL ODE_CONTEXT RELATIVE_TIME
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ -z "${TMUX:-}" && "${CGE_AUTO_TMUX:-1}" != "0" && -t 0 && -t 1 ]]; then
  if command -v tmux >/dev/null 2>&1; then
    futures_for_name=("$@")
    if [[ "${#futures_for_name[@]}" -eq 0 ]]; then
      futures_for_name=(30 45 60 75 90 120 150 180)
    fi
    TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-issm_future_$(IFS=_; echo "${futures_for_name[*]}")_$(date -u +%Y%m%d_%H%M%S)}"
    tmux_cmd=(env CGE_AUTO_TMUX=0 bash "${SCRIPT_PATH}" "$@")
    printf -v tmux_shell_cmd '%q ' "${tmux_cmd[@]}"
    tmux_shell_cmd+="; status=\$?; printf '\\n[future len ablation finished with exit code %s] Press Ctrl-d to close this tmux session.\\n' \"\$status\"; exec bash"
    echo "Starting tmux session: ${TMUX_SESSION_NAME}"
    exec tmux new-session -s "${TMUX_SESSION_NAME}" -c "${SCRIPT_DIR}/.." "${tmux_shell_cmd}"
  else
    echo "tmux not found; running in the current shell." >&2
  fi
fi

if [[ "$#" -gt 0 ]]; then
  future_lens=("$@")
else
  future_lens=(30 45 60 75 90 120 150 180)
fi

for future_len in "${future_lens[@]}"; do
  FUTURE_LEN="$future_len" require_future
  script="${SCRIPT_DIR}/train_issm_node2_future${future_len}.sh"
  if [[ ! -f "${script}" ]]; then
    echo "Unknown future length or missing script: ${future_len}" >&2
    exit 1
  fi
  echo "Running ${script}"
  bash "${script}"
done
