#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="${SESSION_NAME:-issm_node2}"
NPROC="${NPROC:-4}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/issm.yaml}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/protocols/issm/main.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/node2.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG-configs/runtime/fast.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-issm_node2_${STAMP}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"

run_training() {
  mkdir -p "$LOG_DIR"
  export PYTHONUNBUFFERED=1

  {
    echo "[$(date -u +%F' '%T)] project_root=$PROJECT_ROOT"
    echo "[$(date -u +%F' '%T)] session_name=$SESSION_NAME"
    echo "[$(date -u +%F' '%T)] run_name=$RUN_NAME"
    echo "[$(date -u +%F' '%T)] python_bin=$PYTHON_BIN"
    echo "[$(date -u +%F' '%T)] dataset_config=${DATASET_CONFIG:-<none>}"
    echo "[$(date -u +%F' '%T)] model_config=$MODEL_CONFIG"
    echo "[$(date -u +%F' '%T)] runtime_config=${RUNTIME_CONFIG:-<none>}"
    echo "[$(date -u +%F' '%T)] extra_configs=${EXTRA_CONFIGS:-<none>}"
    echo "[$(date -u +%F' '%T)] train_args=${TRAIN_ARGS:-<none>}"
    echo "[$(date -u +%F' '%T)] log_file=$LOG_FILE"
  } | tee -a "$LOG_FILE"

  "$PYTHON_BIN" -c "import sys; import torch; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "$LOG_FILE"

  local extra_config
  local train_cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nproc_per_node="$NPROC"
    scripts/train.py
    --config configs/default.yaml
    --config "$DATASET_CONFIG"
    --config "$PROTOCOL_CONFIG"
    --config "$MODEL_CONFIG"
  )
  for extra_config in $EXTRA_CONFIGS; do
    train_cmd+=(--config "$extra_config")
  done
  if [[ -n "$RUNTIME_CONFIG" ]]; then
    train_cmd+=(--config "$RUNTIME_CONFIG")
  fi
  train_cmd+=(--run-name "$RUN_NAME")
  for train_arg in $TRAIN_ARGS; do
    train_cmd+=("$train_arg")
  done
  "${train_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
}

if [[ "${1:-}" == "--inside-tmux" ]]; then
  run_training
  exit 0
fi

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

export PROJECT_ROOT SESSION_NAME NPROC DATASET_CONFIG PROTOCOL_CONFIG MODEL_CONFIG RUNTIME_CONFIG EXTRA_CONFIGS TRAIN_ARGS PYTHON_BIN STAMP RUN_NAME LOG_DIR LOG_FILE
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/train_issm.sh' --inside-tmux"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: $LOG_FILE"
echo "Output dir: $PROJECT_ROOT/outputs/$RUN_NAME"
