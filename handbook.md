# Shell Script Handbook

This is the quick memory aid for the `.sh` files in this repo.

## ANUGA Training Scripts

### `train_anuga_node1.sh`

Starts ANUGA training for the `NODE1` model.

It sets:

- tmux session name: `anuga_node1`
- model config: `configs/model_node1.yaml`
- run name prefix: `anuga_node1_...`

Use this when you want to train the NODE1 baseline:

```bash
./train_anuga_node1.sh
```

### `train_anuga_node2.sh`

Starts ANUGA training for the `NODE2` model.

It sets:

- tmux session name: `anuga_node2`
- model config: `configs/model_node2.yaml`
- run name prefix: `anuga_node2_...`

Use this when you want to train the default upgraded NODE2 baseline:

```bash
./train_anuga_node2.sh
```

To run the old-baseline ablation while keeping the same launcher:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml ./train_anuga_node2.sh
```

### `train_anuga_node2_upgrade2.sh`

Starts ANUGA training for the structured NODE2 upgrade-v2 overlay.

It sets:

- tmux session name: `anuga_node2_upgrade2`
- model config: `configs/model_node2.yaml`
- extra config: `configs/model_node2_upgrade2.yaml`
- run name prefix: `anuga_node2_upgrade2_...`

The upgrade-v2 overlay uses the structured latent vector field with branch
normalization and gated fusion:

```yaml
model:
  node2_vector_field_type: structured_v2
  structured_dynamics:
    term_norm: rmsnorm
    fusion: softmax_gated
    gate_init: active_mean
```

Use the no-coupling overlay for the matching ablation:

```bash
EXTRA_CONFIGS=configs/model_node2_upgrade2_no_coupling.yaml ./train_anuga_node2.sh
```

### `train_anuga.sh`

This is the shared training launcher used by `train_anuga_node1.sh` and `train_anuga_node2.sh`.

It starts a detached `tmux` session, checks that Python dependencies import correctly, then runs distributed training with:

```bash
python -m torch.distributed.run scripts/train.py
```

Use this directly only when you want to customize the model config or run name:

```bash
MODEL_CONFIG=configs/model_node2.yaml RUN_NAME=anuga_node2_custom bash train_anuga.sh
```

The shared launcher also accepts space-separated override configs through `EXTRA_CONFIGS`.

Large-mesh CUDA memory knobs:

- `model.decoder_chunk_size` chunks `NODE2`/`NCDE1` decoder rows. Default: `262144`.
- `model.history_encoder.history_transformer_chunk_size` chunks temporal-Transformer node batches. Default: `8192`.
- `model.history_encoder.history_transformer_force_math_sdp: true` avoids fused-attention CUDA launch issues on very large node batches.
- Smaller chunks are safer for memory; larger chunks can be faster if the GPU has room.

## ISSM Training Scripts

### `train_issm_node1.sh`

Starts ISSM training for the `NODE1` model.

It sets:

- tmux session name: `issm_node1`
- dataset config: `configs/issm.yaml`
- model config: `configs/model_node1.yaml`
- run name prefix: `issm_node1_...`

Use this when you want to train the ISSM NODE1 baseline:

```bash
./train_issm_node1.sh
```

### `train_issm_node2.sh`

Starts ISSM training for the `NODE2` model.

It sets:

- tmux session name: `issm_node2`
- dataset config: `configs/issm.yaml`
- model config: `configs/model_node2.yaml`
- run name prefix: `issm_node2_...`

Use this when you want to train the default upgraded ISSM NODE2 baseline:

```bash
./train_issm_node2.sh
```

### `train_issm_node2_upgrade2.sh`

Starts ISSM training for the same structured NODE2 upgrade-v2 overlay used by
ANUGA.

It sets:

- tmux session name: `issm_node2_upgrade2`
- dataset config: `configs/issm.yaml`
- model config: `configs/model_node2.yaml`
- extra config: `configs/model_node2_upgrade2.yaml`
- run name prefix: `issm_node2_upgrade2_...`

Use this when you want the structured v2 latent vector field with the
configured branch normalization and gated fusion:

```bash
./train_issm_node2_upgrade2.sh
```

### `train_issm_ncde1.sh`

Starts ISSM training for the `NCDE1` model.

It sets:

- tmux session name: `issm_ncde1`
- dataset config: `configs/issm.yaml`
- model config: `configs/model_ncde1.yaml`
- run name prefix: `issm_ncde1_...`

Use this when you want to train the ISSM NCDE1 baseline:

```bash
./train_issm_ncde1.sh
```

### `train_issm.sh`

This is the shared ISSM training launcher used by `train_issm_node1.sh`, `train_issm_node2.sh`, and `train_issm_ncde1.sh`.

It starts a detached `tmux` session, checks that Python dependencies import correctly, then runs distributed training with:

```bash
python -m torch.distributed.run scripts/train.py
```

Use this directly only when you want to customize the model config or run name:

```bash
MODEL_CONFIG=configs/model_ncde1.yaml RUN_NAME=issm_ncde1_custom bash train_issm.sh
```

## Shared Training Settings

Useful environment variables:

- `NPROC`: number of distributed processes, default `4`
- `MODEL_CONFIG`: model YAML file
- `RUN_NAME`: output folder name under `outputs/`
- `SESSION_NAME`: tmux session name
- `LOG_DIR`: log directory, default `logs/`
- `TRAIN_ARGS`: extra CLI args passed after `--run-name`, for example `--horizon-curriculum off`

Training horizon controls live in YAML under `training:` and are shared by
ANUGA and ISSM:

```yaml
training:
  train_horizon_min: 24
  train_horizon_max: 120
  train_horizon_curriculum:
    enabled: true
    epochs: 120
    warmup_fractions: [0.40, 0.55, 0.70, 0.85]
```

`train_horizon_max` is the target cap for the run. With the default curriculum,
the trainer ramps the epoch-local cap toward that target for the first 120
epochs, then holds the target cap. Direct `scripts/train.py` runs can override
only the enable flag with `--horizon-curriculum on` or
`--horizon-curriculum off`.

For tmux launchers:

```bash
TRAIN_ARGS="--horizon-curriculum off" ./train_anuga_node2.sh
TRAIN_ARGS="--horizon-curriculum off" ./train_issm_node2.sh
```

## Evaluation Scripts

### `eval_window_tmux.sh`

Runs fixed-window evaluation for an existing checkpoint.

Fixed-window evaluation gives the model fresh ground-truth history for each window, so it measures short-window forecasting quality.

Use this when you want to evaluate an ANUGA checkpoint on train, val, or test windows:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval_window_tmux.sh \
  --checkpoint outputs/<run_name>/best.pt \
  --config configs/ANUGA_History_Scan/base_ANUGA_history4.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node1.yaml \
  --history-len 4 \
  --future-len 16 \
  --split test
```

For ISSM, use the same script but swap in `configs/base_ISSM_history_N.yaml` and `configs/issm.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval_window_tmux.sh \
  --checkpoint outputs/<run_name>/best.pt \
  --config configs/base_ISSM_history_N.yaml \
  --config configs/issm.yaml \
  --config configs/model_node1.yaml \
  --history-len 4 \
  --future-len 16 \
  --split test
```

Common options:

- `--checkpoint`: checkpoint file to evaluate
- `--config`: config files to load
- `--split`: `train`, `val`, or `test`
- `--history-len`: number of past state steps
- `--future-len`: number of future prediction steps
- `--gpu`: GPU id to expose through `CUDA_VISIBLE_DEVICES`
- `--session-name`: custom tmux session name

### `eval_full_rollout_tmux.sh`

Runs full-rollout evaluation for an existing checkpoint.

Full-rollout evaluation gives the model only the initial history, then asks it to predict the rest of the scenario. This is stricter than fixed-window evaluation.

Use this when you want to test long-horizon behavior on ANUGA:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval_full_rollout_tmux.sh \
  --checkpoint outputs/<run_name>/best.pt \
  --config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node1.yaml \
  --history-len 1 \
  --known-steps 1 \
  --plot-max-lead-step 60 \
  --split test
```

For ISSM, use the same script but swap in `configs/base_ISSM_history_N.yaml` and `configs/issm.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval_full_rollout_tmux.sh \
  --checkpoint outputs/<run_name>/best.pt \
  --config configs/base_ISSM_history_N.yaml \
  --config configs/issm.yaml \
  --config configs/model_node1.yaml \
  --history-len 1 \
  --known-steps 1 \
  --plot-max-lead-step 60 \
  --split test
```

Common options:

- `--checkpoint`: checkpoint file to evaluate
- `--config`: config files to load
- `--split`: `train`, `val`, or `test`
- `--history-len`: number of initial history steps
- `--known-steps`: number of known starting steps used in rollout setup
- `--plot-max-lead-step`: largest lead step to include in plots
- `--gpu`: GPU id to expose through `CUDA_VISIBLE_DEVICES`
- `--session-name`: custom tmux session name

Training-time full-rollout validation uses `evaluation.full_rollout_known_steps` for the same delayed-start behavior. The standalone script keeps `--known-steps` as a CLI flag so one checkpoint can be evaluated with multiple rollout starts.

## Quick Choice Guide

- Train ANUGA NODE1: `./train_anuga_node1.sh`
- Train ANUGA NODE2: `./train_anuga_node2.sh`
- Custom ANUGA training: `bash train_anuga.sh`
- Train ISSM NODE1: `./train_issm_node1.sh`
- Train ISSM NODE2: `./train_issm_node2.sh`
- Train ISSM NCDE1: `./train_issm_ncde1.sh`
- Custom ISSM training: `bash train_issm.sh`
- Evaluate short-window prediction: `bash eval_window_tmux.sh ...`
- Evaluate long rollout prediction: `bash eval_full_rollout_tmux.sh ...`

## Watching Jobs

All these scripts run inside detached `tmux` sessions.

After starting a job, the script prints the session name. Watch it with:

```bash
tmux attach -t <session-name>
```

Logs are written under `logs/`.
