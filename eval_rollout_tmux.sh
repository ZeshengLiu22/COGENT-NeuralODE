#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SESSION_NAME="${SESSION_NAME:-}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-}}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
CHECKPOINT=""
KNOWN_STEPS=""
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
INSIDE_TMUX=0
EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash eval_rollout_tmux.sh --checkpoint outputs/<run>/best.pt [options]

Roll out from the configured start to trajectory end, saving complete artifacts.
The checkpoint fixes the model, history length, and training future length.

Options:
  --known-steps N             Override the absolute rollout start
  --config PATH              Repeatable runtime-only configuration overlay
  --split train|val|test      Default: test
  --device auto|cpu|cuda      Default: cuda
  --data-dir PATH             Relocate the checkpoint's saved scenarios
  --output-dir PATH           Artifact directory
  --amp-mode none|bf16|fp16   Evaluation precision
  --gpu N                    CUDA_VISIBLE_DEVICES
  --session-name NAME         Detached tmux session name
  --inside-tmux              Run directly (also used by sequential sweeps)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --known-steps) KNOWN_STEPS="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --config|--data-dir|--output-dir|--amp-mode|--plot-max-lead-step|--anuga-num-frames|--anuga-output-dir)
      EVAL_ARGS+=("$1" "$2"); shift 2 ;;
    --skip-anuga-export) EVAL_ARGS+=("$1"); shift ;;
    --inside-tmux) INSIDE_TMUX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "--checkpoint must name an existing checkpoint." >&2
  exit 1
fi
SESSION_NAME="${SESSION_NAME:-rollout_$(basename "$(dirname "$CHECKPOINT")")_${SPLIT}_known${KNOWN_STEPS:-config}}"
SESSION_NAME="$(printf '%s' "$SESSION_NAME" | tr -cs '[:alnum:]_.:-' '_')"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}_${STAMP}.log}"
EVAL_ARGS=(--checkpoint "$CHECKPOINT" --split "$SPLIT" --device "$DEVICE" "${EVAL_ARGS[@]}")
if [[ -n "$KNOWN_STEPS" ]]; then EVAL_ARGS+=(--known-steps "$KNOWN_STEPS"); fi

if [[ "$INSIDE_TMUX" == 1 ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  export PYTHONUNBUFFERED=1
  if [[ "$DEVICE" == cuda* && -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
  "$PYTHON_BIN" scripts/evaluate.py "${EVAL_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2; exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists; attach with tmux attach -t $SESSION_NAME" >&2
  exit 1
fi
tmux_args=(env "PYTHON_BIN=$PYTHON_BIN" "LOG_FILE=$LOG_FILE" "GPU=$GPU"
  bash "$PROJECT_ROOT/eval_rollout_tmux.sh" --inside-tmux --session-name "$SESSION_NAME" "${EVAL_ARGS[@]}")
printf -v tmux_command '%q ' "${tmux_args[@]}"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" "$tmux_command"
echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: $LOG_FILE"
