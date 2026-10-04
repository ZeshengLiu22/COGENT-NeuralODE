#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SCRIPT_PATH="$PROJECT_ROOT/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
SESSION_NAME="${SESSION_NAME:-}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
INSIDE_TMUX=0
CHECKPOINT=""
HISTORY_LEN=""
KNOWN_STEPS=""
PLOT_MAX_LEAD_STEP=""

usage() {
  cat <<'EOF'
Usage:
	  bash eval_full_rollout_tmux.sh \
	    --checkpoint outputs/<run_name>/best.pt \
    [--split test] \
    [--device cuda] \
    [--history-len 8] \
    [--known-steps 8] \
    [--plot-max-lead-step 60] \
    [--gpu 0] \
    [--session-name rollout_eval]

Notes:
	  - The wrapper starts a detached tmux session and writes a timestamped log under logs/.
  - Evaluation uses the config and split saved with the checkpoint.
	  - `--gpu` sets CUDA_VISIBLE_DEVICES for the launched evaluation.
	  - Use `tmux attach -t <session-name>` to watch the run.
EOF
}

sanitize_session_name() {
  printf '%s' "$1" | tr -cs '[:alnum:]_.:-' '_'
}

default_session_name() {
  local checkpoint_dir
  local checkpoint_base
  checkpoint_dir="$(basename "$(dirname "$CHECKPOINT")")"
  checkpoint_base="$(basename "$CHECKPOINT")"
  checkpoint_base="${checkpoint_base%.*}"
  sanitize_session_name "rollout_${checkpoint_dir}_${checkpoint_base}_${SPLIT}"
}

run_eval() {
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}_${STAMP}.log}"
  mkdir -p "$(dirname "$LOG_FILE")"
  export PYTHONUNBUFFERED=1

  if [[ "$DEVICE" == cuda* && -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
  fi

  {
    echo "[$(date -u +%F' '%T)] project_root=$PROJECT_ROOT"
    echo "[$(date -u +%F' '%T)] session_name=$SESSION_NAME"
    echo "[$(date -u +%F' '%T)] checkpoint=$CHECKPOINT"
    echo "[$(date -u +%F' '%T)] split=$SPLIT"
    echo "[$(date -u +%F' '%T)] device=$DEVICE"
    echo "[$(date -u +%F' '%T)] history_len=${HISTORY_LEN:-<config>}"
    echo "[$(date -u +%F' '%T)] known_steps=${KNOWN_STEPS:-<history_len>}"
    echo "[$(date -u +%F' '%T)] plot_max_lead_step=${PLOT_MAX_LEAD_STEP:-<all>}"
    echo "[$(date -u +%F' '%T)] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[$(date -u +%F' '%T)] python_bin=$PYTHON_BIN"
    echo "[$(date -u +%F' '%T)] log_file=$LOG_FILE"
  } | tee -a "$LOG_FILE"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"
  fi

  "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
import sys

import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("cuda_visible_devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
if torch.cuda.is_available():
    current = torch.cuda.current_device()
    print("cuda_current_device:", current)
    print("cuda_device_name:", torch.cuda.get_device_name(current))
PY

  local cmd=("$PYTHON_BIN" "scripts/run_full_rollout.py" "--checkpoint" "$CHECKPOINT" "--split" "$SPLIT" "--device" "$DEVICE")
  if [[ -n "$HISTORY_LEN" ]]; then
    cmd+=("--history-len" "$HISTORY_LEN")
  fi
  if [[ -n "$KNOWN_STEPS" ]]; then
    cmd+=("--known-steps" "$KNOWN_STEPS")
  fi
  if [[ -n "$PLOT_MAX_LEAD_STEP" ]]; then
    cmd+=("--plot-max-lead-step" "$PLOT_MAX_LEAD_STEP")
  fi
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --history-len)
      HISTORY_LEN="$2"
      shift 2
      ;;
    --known-steps)
      KNOWN_STEPS="$2"
      shift 2
      ;;
    --plot-max-lead-step)
      PLOT_MAX_LEAD_STEP="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --inside-tmux)
      INSIDE_TMUX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

SESSION_NAME="${SESSION_NAME:-$(default_session_name)}"
SESSION_NAME="$(sanitize_session_name "$SESSION_NAME")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}_${STAMP}.log}"

if [[ "$INSIDE_TMUX" -eq 1 ]]; then
  run_eval
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

TMUX_CMD=""
printf -v TMUX_CMD "cd %q && env PYTHON_BIN=%q SESSION_NAME=%q LOG_DIR=%q LOG_FILE=%q STAMP=%q DEVICE=%q" \
  "$PROJECT_ROOT" "$PYTHON_BIN" "$SESSION_NAME" "$LOG_DIR" "$LOG_FILE" "$STAMP" "$DEVICE"
if [[ "$DEVICE" == cuda* && -n "$GPU" ]]; then
  printf -v TMUX_CMD "%s CUDA_VISIBLE_DEVICES=%q GPU=%q" "$TMUX_CMD" "$GPU" "$GPU"
fi
printf -v TMUX_CMD "%s bash %q --inside-tmux --checkpoint %q --split %q" \
  "$TMUX_CMD" "$SCRIPT_PATH" "$CHECKPOINT" "$SPLIT"
if [[ -n "$HISTORY_LEN" ]]; then
  printf -v TMUX_CMD "%s --history-len %q" "$TMUX_CMD" "$HISTORY_LEN"
fi
if [[ -n "$KNOWN_STEPS" ]]; then
  printf -v TMUX_CMD "%s --known-steps %q" "$TMUX_CMD" "$KNOWN_STEPS"
fi
if [[ -n "$PLOT_MAX_LEAD_STEP" ]]; then
  printf -v TMUX_CMD "%s --plot-max-lead-step %q" "$TMUX_CMD" "$PLOT_MAX_LEAD_STEP"
fi

tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT"
