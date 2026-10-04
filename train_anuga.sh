#!/usr/bin/env bash
# Ad-hoc single run. Formal experiments use their standalone launchers/ files.
# Defaults: ANUGA, H1/K64/S8, full architecture, TC0, four ranks, trajectory cache on.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NPROC="${NPROC:-4}"
DEFAULT_CONFIG="configs/default.yaml"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/anuga.yaml}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/protocols/anuga/main.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/node2.yaml}"
HISTORY_CONFIG="${HISTORY_CONFIG:-configs/ablations/anuga/history/h1.yaml}"
ARCHITECTURE_CONFIG="${ARCHITECTURE_CONFIG:-configs/ablations/anuga/architecture/full.yaml}"
TRAINING_HORIZON_CONFIG="${TRAINING_HORIZON_CONFIG:-configs/ablations/anuga/training_horizon/k64.yaml}"
ROLLOUT_START_CONFIG="${ROLLOUT_START_CONFIG:-configs/ablations/anuga/rollout_start/known8.yaml}"
TEMPORAL_CONSISTENCY_CONFIG="${TEMPORAL_CONSISTENCY_CONFIG:-configs/ablations/anuga/temporal_consistency/tc0.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG-configs/runtime/anuga_fast.yaml}"
RUN_NAME="${RUN_NAME:-anuga_node2_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/$RUN_NAME.log}"
mkdir -p "$(dirname "$LOG_FILE")"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

{
  echo "project_root=$PROJECT_ROOT"
  echo "python_bin=$PYTHON_BIN"
  echo "nproc=$NPROC"
  echo "run_name=$RUN_NAME"
  echo "output_dir=$PROJECT_ROOT/outputs/$RUN_NAME"
  echo "log_file=$LOG_FILE"
  echo "default_config=$DEFAULT_CONFIG"
  echo "dataset_config=$DATASET_CONFIG"
  echo "protocol_config=$PROTOCOL_CONFIG"
  echo "model_config=$MODEL_CONFIG"
  echo "history_config=$HISTORY_CONFIG"
  echo "architecture_config=$ARCHITECTURE_CONFIG"
  echo "training_horizon_config=$TRAINING_HORIZON_CONFIG"
  echo "rollout_start_config=$ROLLOUT_START_CONFIG"
  echo "temporal_consistency_config=$TEMPORAL_CONSISTENCY_CONFIG"
  if [[ -n "$RUNTIME_CONFIG" ]]; then echo "runtime_config=$RUNTIME_CONFIG"; fi
} | tee -a "$LOG_FILE"

train_cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone --nproc_per_node="$NPROC"
  scripts/train.py
  --config "$DEFAULT_CONFIG"
  --config "$DATASET_CONFIG"
  --config "$PROTOCOL_CONFIG"
  --config "$MODEL_CONFIG"
  --config "$HISTORY_CONFIG"
  --config "$ARCHITECTURE_CONFIG"
  --config "$TRAINING_HORIZON_CONFIG"
  --config "$ROLLOUT_START_CONFIG"
  --config "$TEMPORAL_CONSISTENCY_CONFIG"
)
if [[ -n "$RUNTIME_CONFIG" ]]; then train_cmd+=(--config "$RUNTIME_CONFIG"); fi
train_cmd+=(--run-name "$RUN_NAME")
"${train_cmd[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
