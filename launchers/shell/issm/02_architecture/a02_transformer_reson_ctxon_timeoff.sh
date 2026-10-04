#!/usr/bin/env bash
# ISSM phase 02: a02_transformer_reson_ctxon_timeoff; H=selected after Phase 1, K=180, S=60.
# Architecture: transformer, residual ON, ODE history ON, relative time OFF.
# TC0; fixed training series/scenario/epoch = 60.
# Phases 2–4 propagate only selected H; K/TC scans always use full architecture.
set -euo pipefail

HISTORY_CONFIG="__SET_SELECTED_HISTORY_AFTER_PHASE1__"
if [[ "$HISTORY_CONFIG" == "__SET_SELECTED_HISTORY_AFTER_PHASE1__" ]]; then
  echo "Set HISTORY_CONFIG to the selected Phase-1 history YAML before running." >&2
  exit 1
fi
if [[ ! "$HISTORY_CONFIG" =~ ^configs/ablations/issm/history/h[1-8]\.yaml$ ]]; then
  echo "HISTORY_CONFIG must name this dataset's selected h1 through h8 overlay." >&2
  exit 1
fi
HISTORY_TAG="${HISTORY_CONFIG##*/}"
HISTORY_TAG="${HISTORY_TAG%.yaml}"

PROJECT_ROOT="${PROJECT_ROOT:-/home1/09575/zeshengliu/scratch/COGENT-NeuralODE}"
cd "$PROJECT_ROOT"
# Direct interpreter invocation uses the established torchpyg-cu128-cge environment.
PYTHON_BIN="${PYTHON_BIN:-/work/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python}"
NPROC="${NPROC:-4}"
DEFAULT_CONFIG="configs/default.yaml"
DATASET_CONFIG="configs/datasets/issm.yaml"
PROTOCOL_CONFIG="configs/protocols/issm/main.yaml"
MODEL_CONFIG="configs/models/node2.yaml"
ARCHITECTURE_CONFIG="configs/ablations/issm/architecture/a02_transformer_reson_ctxon_timeoff.yaml"
TRAINING_HORIZON_CONFIG="configs/ablations/issm/training_horizon/k180.yaml"
ROLLOUT_START_CONFIG="configs/ablations/issm/rollout_start/known60.yaml"
TEMPORAL_CONSISTENCY_CONFIG="configs/ablations/issm/temporal_consistency/tc0.yaml"
RUNTIME_CONFIG="configs/runtime/issm_fast.yaml"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="issm_02_architecture_${HISTORY_TAG}_k180_a02_transformer_reson_ctxon_timeoff_tc0_${RUN_STAMP}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$RUN_NAME"
LOG_FILE="$PROJECT_ROOT/logs/$RUN_NAME.log"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

{
  echo "project_root=$PROJECT_ROOT"
  echo "python_bin=$PYTHON_BIN"
  echo "nproc=$NPROC"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<scheduler/default>}"
  echo "slurm_job_id=${SLURM_JOB_ID:-<none>}"
  echo "run_name=$RUN_NAME"
  echo "output_dir=$OUTPUT_DIR"
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
  echo "runtime_config=$RUNTIME_CONFIG"
} | tee -a "$LOG_FILE"

"$PYTHON_BIN" -c "import sys, torch; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())" 2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train.py \
  --config "$DEFAULT_CONFIG" \
  --config "$DATASET_CONFIG" \
  --config "$PROTOCOL_CONFIG" \
  --config "$MODEL_CONFIG" \
  --config "$HISTORY_CONFIG" \
  --config "$ARCHITECTURE_CONFIG" \
  --config "$TRAINING_HORIZON_CONFIG" \
  --config "$ROLLOUT_START_CONFIG" \
  --config "$TEMPORAL_CONSISTENCY_CONFIG" \
  --config "$RUNTIME_CONFIG" \
  --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG_FILE"
