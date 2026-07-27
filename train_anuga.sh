#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="${SESSION_NAME:-anuga_node1}"
NPROC="${NPROC:-4}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_node1.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
BASE_CONFIG="${BASE_CONFIG:-configs/ANUGA_History_Scan/base_ANUGA_history1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-anuga_node1_${STAMP}}"
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
    echo "[$(date -u +%F' '%T)] base_config=$BASE_CONFIG"
    echo "[$(date -u +%F' '%T)] model_config=$MODEL_CONFIG"
    echo "[$(date -u +%F' '%T)] extra_configs=${EXTRA_CONFIGS:-<none>}"
    echo "[$(date -u +%F' '%T)] train_args=${TRAIN_ARGS:-<none>}"
    echo "[$(date -u +%F' '%T)] log_file=$LOG_FILE"
  } | tee -a "$LOG_FILE"

  "$PYTHON_BIN" -c "import sys; import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "$LOG_FILE"

  if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Model config not found: $MODEL_CONFIG" >&2
    exit 1
  fi
  local extra_config
  for extra_config in $EXTRA_CONFIGS; do
    if [[ ! -f "$extra_config" ]]; then
      echo "Extra config not found: $extra_config" >&2
      exit 1
    fi
  done

  local train_cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nproc_per_node="$NPROC"
    scripts/train.py
    --config "$BASE_CONFIG"
    --config configs/anuga.yaml
    --config "$MODEL_CONFIG"
  )
  for extra_config in $EXTRA_CONFIGS; do
    train_cmd+=(--config "$extra_config")
  done
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

"$PYTHON_BIN" -c "import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy" >/dev/null

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

export PROJECT_ROOT SESSION_NAME NPROC MODEL_CONFIG EXTRA_CONFIGS TRAIN_ARGS BASE_CONFIG PYTHON_BIN STAMP RUN_NAME LOG_DIR LOG_FILE
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/train_anuga.sh' --inside-tmux"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: $LOG_FILE"
echo "Output dir: $PROJECT_ROOT/outputs/$RUN_NAME"
