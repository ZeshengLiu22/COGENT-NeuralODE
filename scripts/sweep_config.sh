#!/usr/bin/env bash
# Shared experiment definitions; this file schedules no jobs and starts no training.

require_history() {
  if [[ ! "${HISTORY_LEN:-}" =~ ^[1-8]$ ]]; then
    echo "Set HISTORY_LEN to the selected history (1 through 8)." >&2
    return 1
  fi
}

require_architecture() {
  if [[ "${ENCODER:-}" != transformer && "${ENCODER:-}" != lstm ]]; then
    echo "Set ENCODER to the selected transformer or lstm architecture." >&2
    return 1
  fi
  local axis
  for axis in RESIDUAL ODE_CONTEXT RELATIVE_TIME; do
    if [[ "${!axis:-}" != on && "${!axis:-}" != off ]]; then
      echo "Set ${axis} to the selected on or off setting." >&2
      return 1
    fi
  done
}

require_future() {
  case "${FUTURE_LEN:-}" in
    30|45|60|75|90|120|150|180) ;;
    *) echo "Set FUTURE_LEN to 30, 45, 60, 75, 90, 120, 150, or 180." >&2; return 1 ;;
  esac
}

configure_sweep() {
  local dataset="$1" phase="$2" known tc
  require_history || return
  known=60
  tc=tc0_off
  case "$phase" in
    history)
      FUTURE_LEN=180
      [[ "$dataset" != anuga ]] || { FUTURE_LEN=64; known=8; }
      ENCODER=transformer RESIDUAL=on ODE_CONTEXT=on RELATIVE_TIME=on
      ;;
    architecture)
      require_architecture || return
      FUTURE_LEN=180
      ;;
    future)
      require_architecture || return
      require_future || return
      ;;
    temporal_consistency)
      require_architecture || return
      require_future || return
      case "${TC_MODE:-}" in
        tc0_off|tc1_adjacent_increment|tc2_random_pair_increment|tc3_multiscale_rate|tc4_rate_curvature|tc5_hybrid) tc="$TC_MODE" ;;
        *) echo "Set TC_MODE to one of the six temporal consistency config names." >&2; return 1 ;;
      esac
      ;;
    *) echo "Unknown sweep phase: $phase" >&2; return 1 ;;
  esac
  SWEEP_CONFIGS=(
    "configs/ablations/history/h${HISTORY_LEN}.yaml"
    "configs/ablations/future_len/k${FUTURE_LEN}.yaml"
    "configs/ablations/rollout_start/known${known}.yaml"
    "configs/ablations/architecture/encoder/${ENCODER}.yaml"
    "configs/ablations/architecture/residual/${RESIDUAL}.yaml"
    "configs/ablations/architecture/ode_context/${ODE_CONTEXT}.yaml"
    "configs/ablations/architecture/relative_time/${RELATIVE_TIME}.yaml"
    "configs/ablations/temporal_consistency/${tc}.yaml"
  )
  SWEEP_LABEL="h${HISTORY_LEN}_k${FUTURE_LEN}_${ENCODER}_res${RESIDUAL}_ctx${ODE_CONTEXT}_time${RELATIVE_TIME}_${tc}"
}
