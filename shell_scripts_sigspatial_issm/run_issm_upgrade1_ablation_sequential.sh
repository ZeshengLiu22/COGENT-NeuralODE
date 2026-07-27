#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

all_ablations=(
  upgrade_off
  residual_only
  history_in_ode_only
  relative_time_only
  no_residual
  no_history_in_ode
  no_relative_time
  full_upgrade
)

if [[ -z "${TMUX:-}" && "${CGE_AUTO_TMUX:-1}" != "0" && -t 0 && -t 1 ]]; then
  if command -v tmux >/dev/null 2>&1; then
    ablations_for_name=("$@")
    if [[ "${#ablations_for_name[@]}" -eq 0 ]]; then
      ablations_for_name=(all)
    fi
    TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-issm_ablation_$(IFS=_; echo "${ablations_for_name[*]}")_$(date -u +%Y%m%d_%H%M%S)}"
    tmux_cmd=(env CGE_AUTO_TMUX=0 bash "${SCRIPT_PATH}" "$@")
    printf -v tmux_shell_cmd '%q ' "${tmux_cmd[@]}"
    tmux_shell_cmd+="; status=\$?; printf '\\n[ablation scan finished with exit code %s] Press Ctrl-d to close this tmux session.\\n' \"\$status\"; exec bash"
    echo "Starting tmux session: ${TMUX_SESSION_NAME}"
    exec tmux new-session -s "${TMUX_SESSION_NAME}" -c "${SCRIPT_DIR}/.." "${tmux_shell_cmd}"
  else
    echo "tmux not found; running in the current shell." >&2
  fi
fi

if [[ "$#" -gt 0 ]]; then
  ablations=("$@")
else
  ablations=("${all_ablations[@]}")
fi

for ablation in "${ablations[@]}"; do
  script="${SCRIPT_DIR}/train_issm_node2_ablate_${ablation}.sh"
  if [[ ! -f "${script}" ]]; then
    echo "Unknown ablation or missing script: ${ablation}" >&2
    exit 1
  fi
  echo "Running ${script} with HISTORY_LEN=${HISTORY_LEN:-4}"
  bash "${script}"
done
