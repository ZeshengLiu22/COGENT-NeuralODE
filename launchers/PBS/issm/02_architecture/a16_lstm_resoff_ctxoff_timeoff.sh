#!/bin/bash -l
#PBS -N cogent_a100
#PBS -A ULHI0006
#PBS -q casper
#PBS -l select=1:ncpus=16:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100_80gb
#PBS -l walltime=12:00:00
#PBS -j oe

# One A100 80GB; torchrun starts one rank and the final YAML accumulates four batches.
# ISSM phase 02: a16_lstm_resoff_ctxoff_timeoff; H=selected after Phase 1, K=180, S=60.
# Architecture: lstm, residual OFF, ODE history OFF, relative time OFF.
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

PROJECT_ROOT="${PROJECT_ROOT:-/glade/u/home/zel/scratch/COGENT-NeuralODE}"
cd "$PROJECT_ROOT"
# Direct interpreter invocation uses the Casper casper-ml environment.
PYTHON_BIN="${PYTHON_BIN:-/glade/work/zel/conda-envs/casper-ml/bin/python}"
NPROC=1
DEFAULT_CONFIG="configs/default.yaml"
DATASET_CONFIG="configs/datasets/issm.yaml"
PROTOCOL_CONFIG="configs/protocols/issm/main.yaml"
MODEL_CONFIG="configs/models/node2.yaml"
ARCHITECTURE_CONFIG="configs/ablations/issm/architecture/a16_lstm_resoff_ctxoff_timeoff.yaml"
TRAINING_HORIZON_CONFIG="configs/ablations/issm/training_horizon/k180.yaml"
ROLLOUT_START_CONFIG="configs/ablations/issm/rollout_start/known60.yaml"
TEMPORAL_CONSISTENCY_CONFIG="configs/ablations/issm/temporal_consistency/tc0.yaml"
RUNTIME_CONFIG="configs/runtime/issm_fast.yaml"
FINAL_CONFIG="configs/runtime/single_a100.yaml"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="issm_02_architecture_${HISTORY_TAG}_k180_a16_lstm_resoff_ctxoff_timeoff_tc0_${RUN_STAMP}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$RUN_NAME"
LOG_FILE="$PROJECT_ROOT/logs/$RUN_NAME.log"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

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
  echo "pbs_job_id=${PBS_JOBID:-<none>}"
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
  echo "final_config=$FINAL_CONFIG"
} | tee -a "$LOG_FILE"

"$PYTHON_BIN" -c "import sys, torch; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())" 2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
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
  --config "$FINAL_CONFIG" \
  --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG_FILE"
