#!/bin/bash -l
#PBS -N issm_01_h1
#PBS -A ULHI0006
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=64:mpiprocs=4:ompthreads=1:ngpus=4
#PBS -l place=excl
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe
#PBS -M zel220@lehigh.edu

# One exclusive Derecho GPU node; torchrun starts four ranks with accumulation 2.
# ISSM phase 01: h1; H=1, K=180, S=60.
# Architecture: transformer, residual ON, ODE history ON, relative time ON.
# TC0; fixed training series/scenario/epoch = 60.
# Phases 2–4 propagate only selected H; K/TC scans always use full architecture.
set -euo pipefail

HISTORY_CONFIG="configs/ablations/issm/history/h1.yaml"
if [[ ! "$HISTORY_CONFIG" =~ ^configs/ablations/issm/history/h[1-8]\.yaml$ ]]; then
  echo "HISTORY_CONFIG must name this dataset's selected h1 through h8 overlay." >&2
  exit 1
fi
HISTORY_TAG="${HISTORY_CONFIG##*/}"
HISTORY_TAG="${HISTORY_TAG%.yaml}"

PROJECT_ROOT="${PROJECT_ROOT:-/glade/u/home/zel/scratch/COGENT-NeuralODE}"
cd "$PROJECT_ROOT"
# Use the existing Derecho environment and initialize the node's CUDA support.
module load cuda
PYTHON_BIN="${PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}"
NPROC=4
DEFAULT_CONFIG="configs/default.yaml"
DATASET_CONFIG="configs/datasets/issm.yaml"
PROTOCOL_CONFIG="configs/protocols/issm/main.yaml"
MODEL_CONFIG="configs/models/node2.yaml"
ARCHITECTURE_CONFIG="configs/ablations/issm/architecture/full.yaml"
TRAINING_HORIZON_CONFIG="configs/ablations/issm/training_horizon/k180.yaml"
ROLLOUT_START_CONFIG="configs/ablations/issm/rollout_start/known60.yaml"
TEMPORAL_CONSISTENCY_CONFIG="configs/ablations/issm/temporal_consistency/tc0.yaml"
RUNTIME_CONFIG="configs/runtime/issm_fast.yaml"
FINAL_CONFIG="configs/runtime/issm_derecho.yaml"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S_%N)}"
RUN_NAME="issm_01_history_${HISTORY_TAG}_k180_full_tc0_${RUN_STAMP}_pbs${PBS_JOBID:-local_$$}"
RUN_DIR="$PROJECT_ROOT/outputs/$RUN_NAME"
TRAIN_DIR="$RUN_DIR/train"
INFER_DIR="$RUN_DIR/inference"
LOG_FILE="$TRAIN_DIR/launcher.log"
INFER_LOG="$INFER_DIR/inference.log"
METADATA_FILE="$TRAIN_DIR/runtime_metadata.txt"
mkdir -p "$PROJECT_ROOT/outputs"
# Reserve a new folder atomically; never reuse or overwrite an earlier run.
mkdir "$RUN_DIR"
mkdir "$TRAIN_DIR" "$INFER_DIR"

# Metadata is best effort; unavailable tools or metadata writes must not stop a run.
record_metadata() {
  printf '%s\n' "$@" | tee -a "$LOG_FILE" "$METADATA_FILE" || true
}
utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf 'unavailable'
}

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
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')"
  echo "gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || printf 'unavailable')"
  echo "run_name=$RUN_NAME"
  echo "output_dir=$RUN_DIR"
  echo "train_dir=$TRAIN_DIR"
  echo "inference_dir=$INFER_DIR"
  echo "inference_log=$INFER_LOG"
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
} | tee -a "$LOG_FILE" "$METADATA_FILE" || true

"$PYTHON_BIN" -c "import sys, torch; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())" 2>&1 | tee -a "$LOG_FILE"

record_metadata "training_start_utc=$(utc_now)"
if "$PYTHON_BIN" -m torch.distributed.run \
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
  --run-name "$RUN_NAME/train" 2>&1 | tee -a "$LOG_FILE"; then
  record_metadata "training_end_utc=$(utc_now)" "training_exit_status=0"
else
  status=$?
  record_metadata "training_end_utc=$(utc_now)" "training_exit_status=$status"
  exit "$status"
fi

if [[ ! -f "$TRAIN_DIR/best.pt" || ! -s "$TRAIN_DIR/best.pt" ]]; then
  echo "Best checkpoint is missing or empty: $TRAIN_DIR/best.pt" | tee -a "$LOG_FILE" >&2
  exit 1
fi

record_metadata "inference_start_utc=$(utc_now)"
if "$PYTHON_BIN" scripts/evaluate.py \
  --checkpoint "$TRAIN_DIR/best.pt" \
  --split test \
  --device cuda \
  --amp-mode none \
  --output-dir "$INFER_DIR" 2>&1 | tee -a "$INFER_LOG"; then
  record_metadata "inference_end_utc=$(utc_now)" "inference_exit_status=0"
else
  status=$?
  record_metadata "inference_end_utc=$(utc_now)" "inference_exit_status=$status"
  exit "$status"
fi
