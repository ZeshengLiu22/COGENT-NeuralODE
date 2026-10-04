#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ -z "${TMUX:-}" && "${CGE_AUTO_TMUX:-1}" != "0" && -t 0 && -t 1 ]]; then
  if command -v tmux >/dev/null 2>&1; then
    histories_for_name=("$@")
    if [[ "${#histories_for_name[@]}" -eq 0 ]]; then
      histories_for_name=(1 2 3 4 5 6 7 8)
    fi
    TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-issm_history_$(IFS=_; echo "${histories_for_name[*]}")_$(date -u +%Y%m%d_%H%M%S)}"
    tmux_cmd=(env CGE_AUTO_TMUX=0 bash "${SCRIPT_PATH}" "$@")
    printf -v tmux_shell_cmd '%q ' "${tmux_cmd[@]}"
    tmux_shell_cmd+="; status=\$?; printf '\\n[history scan finished with exit code %s] Press Ctrl-d to close this tmux session.\\n' \"\$status\"; exec bash"
    echo "Starting tmux session: ${TMUX_SESSION_NAME}"
    exec tmux new-session -s "${TMUX_SESSION_NAME}" -c "${SCRIPT_DIR}/.." "${tmux_shell_cmd}"
  else
    echo "tmux not found; running in the current shell." >&2
  fi
fi

if [[ "$#" -gt 0 ]]; then
  histories=("$@")
else
  histories=(1 2 3 4 5 6 7 8)
fi

for history_len in "${histories[@]}"; do
  if [[ ! "${history_len}" =~ ^[1-8]$ ]]; then
    echo "History length must be an integer from 1 to 8, got: ${history_len}" >&2
    exit 1
  fi
  script="${SCRIPT_DIR}/train_issm_node2_history${history_len}_full_upgrade1_1.sh"
  echo "Running ${script}"
  bash "${script}"
done
