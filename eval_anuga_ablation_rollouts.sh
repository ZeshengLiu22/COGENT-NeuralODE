#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HISTORY_LEN="${HISTORY_LEN:-1}"
KNOWN_STEPS="${KNOWN_STEPS:-1}"
PLOT_MAX_LEAD_STEP="${PLOT_MAX_LEAD_STEP:-60}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
BASE_CONFIG="${BASE_CONFIG:-configs/ANUGA_History_Scan/base_ANUGA_history1.yaml}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-anuga_ablation_rollouts_${STAMP}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}.log}"
INSIDE_TMUX=0
DRY_RUN=0

ABLATION_NAMES=(
  full_upgrade
  upgrade_off
  transformer_history_only
  residual_only
  relative_time_only
  history_in_ode_only
)

declare -A RUN_PREFIXES=(
  [full_upgrade]="anuga_node2_full_upgrade"
  [upgrade_off]="anuga_node2_upgrade_off"
  [transformer_history_only]="anuga_node2_transformer_history_only"
  [residual_only]="anuga_node2_residual_only"
  [relative_time_only]="anuga_node2_relative_time_only"
  [history_in_ode_only]="anuga_node2_history_in_ode_only"
)

declare -A EXTRA_CONFIGS=(
  [full_upgrade]=""
  [upgrade_off]="configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml"
  [transformer_history_only]="configs/NODE2_Upgrade1_Ablation/model_node2_transformer_history_only.yaml"
  [residual_only]="configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml"
  [relative_time_only]="configs/NODE2_Upgrade1_Ablation/model_node2_relative_time_only.yaml"
  [history_in_ode_only]="configs/NODE2_Upgrade1_Ablation/model_node2_history_in_ode_only.yaml"
)

usage() {
  cat <<'EOF'
Usage:
  bash eval_anuga_ablation_rollouts.sh [options]

Runs the six ANUGA NODE2 ablation checkpoints sequentially in one tmux session.
By default each checkpoint is the latest matching:
  outputs/anuga_node2_<ablation>_*/best.pt

Options:
  --split train|val|test           Dataset split to evaluate. Default: test
  --gpu 0                          GPU id for CUDA_VISIBLE_DEVICES. Default: 0
  --device cuda|cpu|auto           Evaluation device. Default: cuda
  --history-len 1                  Evaluation history length. Default: 1
  --known-steps 1                  Known rollout steps. Default: 1
  --plot-max-lead-step 60          Lead steps saved in plots/summary. Default: 60
  --session-name NAME              tmux session name
  --dry-run                        Print commands without running them
  -h, --help                       Show this help

Environment overrides:
	  PYTHON_BIN, SPLIT, DEVICE, GPU, HISTORY_LEN, KNOWN_STEPS,
	  PLOT_MAX_LEAD_STEP, LOG_DIR, BASE_CONFIG, SESSION_NAME, LOG_FILE
EOF
}

sanitize_session_name() {
  printf '%s' "$1" | tr -cs '[:alnum:]_.:-' '_'
}

latest_checkpoint_for() {
  local prefix="$1"
  local latest=""
  local candidate
  shopt -s nullglob
  for candidate in "$PROJECT_ROOT"/outputs/"${prefix}"_*/best.pt; do
    latest="$candidate"
  done
  shopt -u nullglob
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

append_config_args() {
  local -n out_args="$1"
  local extra_config="$2"
  out_args+=(--config "$BASE_CONFIG")
  out_args+=(--config configs/anuga.yaml)
  out_args+=(--config configs/model_node2.yaml)
  if [[ -n "$extra_config" ]]; then
    out_args+=(--config "$extra_config")
  fi
}

run_one_rollout() {
  local ablation_name="$1"
  local checkpoint="$2"
  local extra_config="${EXTRA_CONFIGS[$ablation_name]}"
  local run_name
  run_name="$(basename "$(dirname "$checkpoint")")"
  local run_log="$LOG_DIR/${run_name}.full_rollout_${SPLIT}_${STAMP}.log"
  local -a cmd=(
    "$PYTHON_BIN" scripts/run_full_rollout.py
    --checkpoint "$checkpoint"
    --split "$SPLIT"
    --device "$DEVICE"
    --history-len "$HISTORY_LEN"
    --known-steps "$KNOWN_STEPS"
    --plot-max-lead-step "$PLOT_MAX_LEAD_STEP"
  )
  append_config_args cmd "$extra_config"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "[$(date -u +%F' '%T)] ===== ${ablation_name} ====="
    echo "[$(date -u +%F' '%T)] checkpoint=${checkpoint}"
    echo "[$(date -u +%F' '%T)] run_log=${run_log}"
    printf '[%s] command=' "$(date -u +%F' '%T)"
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  {
    echo
    echo "[$(date -u +%F' '%T)] ===== ${ablation_name} ====="
    echo "[$(date -u +%F' '%T)] checkpoint=${checkpoint}"
    echo "[$(date -u +%F' '%T)] run_log=${run_log}"
    printf '[%s] command=' "$(date -u +%F' '%T)"
    printf '%q ' "${cmd[@]}"
    echo
  } | tee -a "$LOG_FILE"

  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE" "$run_log"
}

run_all_rollouts() {
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
  fi
  export PYTHONUNBUFFERED=1
  if [[ "$DEVICE" == cuda* && -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$(date -u +%F' '%T)] project_root=$PROJECT_ROOT"
    echo "[$(date -u +%F' '%T)] session_name=$SESSION_NAME"
    echo "[$(date -u +%F' '%T)] split=$SPLIT"
    echo "[$(date -u +%F' '%T)] device=$DEVICE"
    echo "[$(date -u +%F' '%T)] history_len=$HISTORY_LEN"
    echo "[$(date -u +%F' '%T)] known_steps=$KNOWN_STEPS"
    echo "[$(date -u +%F' '%T)] plot_max_lead_step=$PLOT_MAX_LEAD_STEP"
    echo "[$(date -u +%F' '%T)] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[$(date -u +%F' '%T)] python_bin=$PYTHON_BIN"
    echo "[$(date -u +%F' '%T)] log_file=$LOG_FILE"
    echo "[$(date -u +%F' '%T)] base_config=$BASE_CONFIG"
  else
  {
    echo "[$(date -u +%F' '%T)] project_root=$PROJECT_ROOT"
    echo "[$(date -u +%F' '%T)] session_name=$SESSION_NAME"
    echo "[$(date -u +%F' '%T)] split=$SPLIT"
    echo "[$(date -u +%F' '%T)] device=$DEVICE"
    echo "[$(date -u +%F' '%T)] history_len=$HISTORY_LEN"
    echo "[$(date -u +%F' '%T)] known_steps=$KNOWN_STEPS"
    echo "[$(date -u +%F' '%T)] plot_max_lead_step=$PLOT_MAX_LEAD_STEP"
    echo "[$(date -u +%F' '%T)] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[$(date -u +%F' '%T)] python_bin=$PYTHON_BIN"
	    echo "[$(date -u +%F' '%T)] log_file=$LOG_FILE"
	    echo "[$(date -u +%F' '%T)] base_config=$BASE_CONFIG"
  } | tee -a "$LOG_FILE"

  "$PYTHON_BIN" -c "import sys; import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>&1 | tee -a "$LOG_FILE"
  fi

  local ablation_name
  for ablation_name in "${ABLATION_NAMES[@]}"; do
    local checkpoint
    if ! checkpoint="$(latest_checkpoint_for "${RUN_PREFIXES[$ablation_name]}")"; then
      echo "No checkpoint found for ${ablation_name}: outputs/${RUN_PREFIXES[$ablation_name]}_*/best.pt" >&2
      exit 1
    fi
    run_one_rollout "$ablation_name" "$checkpoint"
  done

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$(date -u +%F' '%T)] Dry run finished."
  else
    echo "[$(date -u +%F' '%T)] All ANUGA ablation rollouts finished." | tee -a "$LOG_FILE"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
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
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "$SPLIT" != "train" && "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
  echo "--split must be one of: train, val, test." >&2
  exit 1
fi

for config_path in "$BASE_CONFIG" configs/anuga.yaml configs/model_node2.yaml "${EXTRA_CONFIGS[@]}"; do
  if [[ -n "$config_path" && ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
  fi
done

SESSION_NAME="$(sanitize_session_name "$SESSION_NAME")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}.log}"

if [[ "$INSIDE_TMUX" -eq 1 ]]; then
  run_all_rollouts
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  run_all_rollouts
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

export PYTHON_BIN SPLIT DEVICE GPU HISTORY_LEN KNOWN_STEPS PLOT_MAX_LEAD_STEP LOG_DIR BASE_CONFIG SESSION_NAME LOG_FILE STAMP
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/eval_anuga_ablation_rollouts.sh' --inside-tmux"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: $LOG_FILE"
