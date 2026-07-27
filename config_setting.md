# Config Setting Guide

This file explains how to set the YAML configs in this repo.

The short version:

```bash
python scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history8.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node1.yaml
```

Config files are merged from left to right. Later files override earlier files.

For example, `configs/ANUGA_History_Scan/base_ANUGA_history8.yaml` contains ANUGA-oriented training defaults with eight history steps, while `configs/base_ISSM_history_N.yaml` contains ISSM-oriented history/window/evaluation defaults.

## Config Files

| File | Purpose |
| --- | --- |
| `configs/ANUGA_History_Scan/base_ANUGA_history<N>.yaml` | ANUGA-oriented defaults for dataset windows, model size, solver, training, evaluation, DDP, and AMP. Choose `<N>` from `1` through `8` to set `dataset.history_len`. |
| `configs/base_ISSM_history_N.yaml` | ISSM-oriented defaults for dataset windows, model size, solver, training, evaluation, DDP, and AMP. |
| `configs/base_sample.yaml` | Small/shared sample defaults retained for quick local runs and compatibility. |
| `configs/anuga.yaml` | ANUGA dataset path and file pattern. |
| `configs/adcirc.yaml` | ADCIRC dataset path and file patterns. |
| `configs/issm.yaml` | ISSM dataset path, file patterns, and default ISSM split rule. |
| `configs/model_node1.yaml` | Selects the `NODE1` model and linear interpolation. |
| `configs/model_node2.yaml` | Selects the upgraded `NODE2` model with residual decoder, history-in-ODE, Transformer history, and relative time enabled. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml` | NODE2 overlay that disables all upgrade-v1 switches and restores the old LSTM/absolute-decoder baseline as closely as possible. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml` | NODE2 overlay for testing only the residual decoder upgrade on top of the old baseline-like mode. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_history_in_ode_only.yaml` | NODE2 overlay for testing only history context inside the ODE vector field. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_transformer_history_only.yaml` | NODE2 overlay for testing only the Transformer history encoder. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_relative_time_only.yaml` | NODE2 overlay for testing only normalized relative time in the ODE vector field. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_residual.yaml` | NODE2 leave-one-out overlay: full upgraded model except residual decoder. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_history_in_ode.yaml` | NODE2 leave-one-out overlay: full upgraded model except history-in-ODE. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_transformer_history.yaml` | NODE2 leave-one-out overlay: full upgraded model except Transformer history, falling back to LSTM. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_relative_time.yaml` | NODE2 leave-one-out overlay: full upgraded model except relative time. |
| `configs/model_ncde1.yaml` | Selects the `NCDE1` model and CDE solver defaults. |

Common combinations:

```bash
# ANUGA NODE1
--config configs/ANUGA_History_Scan/base_ANUGA_history8.yaml --config configs/anuga.yaml --config configs/model_node1.yaml

# ANUGA NODE2
--config configs/ANUGA_History_Scan/base_ANUGA_history8.yaml --config configs/anuga.yaml --config configs/model_node2.yaml

# ANUGA NODE2 old baseline-like ablation
--config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml --config configs/anuga.yaml --config configs/model_node2.yaml --config configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml

# ISSM NODE1
--config configs/base_ISSM_history_N.yaml --config configs/issm.yaml --config configs/model_node1.yaml

# ISSM NODE2
--config configs/base_ISSM_history_N.yaml --config configs/issm.yaml --config configs/model_node2.yaml

# ISSM NODE2 residual-only ablation
--config configs/base_ISSM_history_N.yaml --config configs/issm.yaml --config configs/model_node2.yaml --config configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml

# ISSM NCDE1
--config configs/base_ISSM_history_N.yaml --config configs/issm.yaml --config configs/model_ncde1.yaml

# ADCIRC NODE1
--config configs/base_sample.yaml --config configs/adcirc.yaml --config configs/model_node1.yaml
```

## Merge Rules

- YAML dictionaries are merged recursively.
- Later config files override earlier files.
- Lists are replaced, not appended.
- Unknown keys are usually ignored unless some code path explicitly reads them.
- Missing required keys usually fail at runtime with `KeyError` or `ValueError`.

Example:

```yaml
# configs/ANUGA_History_Scan/base_ANUGA_history8.yaml
model:
  latent_dim: 64
  decoder_hidden_dims: [64, 64]

# custom_override.yaml
model:
  latent_dim: 96
```

Final result:

```yaml
model:
  latent_dim: 96
  decoder_hidden_dims: [64, 64]
```

Because this is recursive merge, `decoder_hidden_dims` stays unless a later file replaces it.

## Top-Level Settings

| Key | Supported values | What it does |
| --- | --- | --- |
| `seed` | integer | Seeds Python, NumPy, and PyTorch. DDP ranks use `seed + rank`. |
| `deterministic` | `true` or `false` | If `true`, sets cuDNN deterministic mode and disables cuDNN benchmark. More reproducible, sometimes slower. |
| `output_dir` | path string | Training writes run folders here, for example `outputs/<run_name>/`. |

## Dataset Settings

These live under `dataset:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `name` | `anuga`, `adcirc`, `issm` | Chooses the dataset adapter. Must match the dataset config you are using. |
| `data_dir` | path string | Directory where scenario files are discovered. |
| `file_patterns` | string list | Glob patterns searched under `data_dir`. Multiple patterns are allowed. |
| `history_len` | integer `>= 1` | Number of past state/forcing steps given to the encoder. Larger values give more context but cost more memory/time. |
| `future_len` | integer `>= 1` | Number of future target steps stored in each training/fixed-window sample. Larger values train/evaluate longer windows. |
| `stride` | integer `>= 1` | Step size between valid windows. `1` gives maximum overlap and most samples. Larger values reduce sample count. |
| `num_workers` | integer `>= 0` | DataLoader worker count for training, validation, and test loaders during training. |
| `cache_in_memory` | `true` or `false` | If `true`, loaded trajectories are cached in RAM. Faster repeated access, higher memory use. |
| `seed` | integer, optional | Dataset/window sampling seed. If absent, top-level `seed` is used. |

### Window Meaning

One fixed-window sample contains:

- `history_len` observed state steps
- `history_len` observed forcing steps
- `future_len` known future forcing steps
- `future_len` future target state steps

Full-rollout evaluation uses `history_len` as the initial context, then predicts the rest of the trajectory.

Standalone evaluation can override these with CLI flags:

- fixed-window: `--history-len`, `--future-len`
- full-rollout: `--history-len`, `--known-steps`

## Dataset Split Settings

These live under `dataset.split:`.

### `strategy: random`

Supported keys:

| Key | Supported values | What it does |
| --- | --- | --- |
| `strategy` | `random` | Shuffles scenario files and splits by fraction. |
| `train` | float | Training fraction. |
| `val` | float | Validation fraction. |
| `test` | float | Test fraction. |
| `seed` | integer | Shuffle seed. If absent, top-level `seed` is used. |

`train + val + test` must equal `1.0`.

Example:

```yaml
dataset:
  split:
    strategy: random
    train: 0.7
    val: 0.15
    test: 0.15
    seed: 42
```

### `strategy: issm_rate_modulo`

Supported keys:

| Key | Supported values | What it does |
| --- | --- | --- |
| `strategy` | `issm_rate_modulo` | Splits ISSM files by melt-rate tags in filenames. |
| `modulo` | integer | Uses `rate % modulo` to assign files. |
| `val_remainder` | integer | Files with this remainder go to validation. |
| `test_remainder` | integer | Files with this remainder go to test. |

This expects ISSM filenames to contain a tag like `_r010`, `_r020`, etc.

Default ISSM split:

```yaml
dataset:
  split:
    strategy: issm_rate_modulo
    modulo: 20
    val_remainder: 0
    test_remainder: 10
```

With that setting:

- rates where `rate % 20 == 0` go to validation
- rates where `rate % 20 == 10` go to test
- all other rates go to training

## Sampled Window Settings

These live under `dataset.sampled_windows:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `windows_per_scenario` | integer or `null` | If set, samples at most this many windows from each training scenario per epoch. |
| `epoch_num_windows` | integer or `null` | If set, caps the total number of sampled training windows per epoch. |

These settings affect only the training split.

If both are `null`, every valid training window is used every epoch.

If trajectories are long, use these to make training epochs shorter:

```yaml
dataset:
  sampled_windows:
    windows_per_scenario: 32
    epoch_num_windows: 2048
```

## Dataset-Specific Files

### ANUGA

Config file:

```yaml
dataset:
  name: anuga
  data_dir: ./data/ANUGA/simulation_data_merged
  file_patterns:
    - "sim_*_merged.npz"
  anuga: {}
```

Supported file type:

- `.npz`

Expected arrays include coordinates/static fields, graph connectivity, time, rain, and state variables. The adapter builds state as:

- `depth`
- `xmomentum`
- `ymomentum`

`dataset.anuga` is currently reserved. It is passed through but no adapter options are read from it yet.

### ADCIRC

Config file:

```yaml
dataset:
  name: adcirc
  data_dir: ./data/adcirc
  file_patterns:
    - "*.npz"
    - "*.pt"
    - "*.pth"
    - "*.mat"
  adcirc: {}
```

Supported file types:

- `.npz`
- `.pt`
- `.pth`
- `.mat`

The adapter accepts generic `state` or `surge` payloads. If state has one channel, it is interpreted as surge.

`dataset.adcirc` is currently reserved. It is passed through but no adapter options are read from it yet.

### ISSM

Config file:

```yaml
dataset:
  name: issm
  data_dir: ./data/ISSM/PIG_data
  file_patterns:
    - "*.mat"
    - "*.npz"
    - "*.pt"
    - "*.pth"
  issm: {}
```

Supported file types:

- legacy `.mat` files with `S`
- generic `.npz`
- generic `.pt`
- generic `.pth`

Legacy ISSM state is:

- `vx`
- `vy`
- `thickness`

Legacy ISSM forcing is:

- melt rate
- surface mass balance
- floating mask/value

`dataset.issm` is currently reserved. It is passed through but no adapter options are read from it yet.

## Normalization Settings

These live under `normalization:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `std_floor` | positive float | Protects against tiny standard deviations when fitting train-split-only normalization. |

The normalizer is fit only on training trajectories.

If a channel standard deviation is smaller than `std_floor`, the code uses standard deviation `1.0` for that channel. That avoids exploding normalized values for nearly constant channels.

## Model Choice

These live under `model:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `name` | `node1`, `node2`, `ncde1` | Selects the model class. |

### `name: node1`

`NODE1` is a state-space controlled Graph Neural ODE.

It evolves the physical state directly:

```text
state -> ODE -> future state
```

Important behavior:

- Uses `model.history_encoder`.
- Uses `model.continuous`.
- Uses `solver.ode_method`.
- Uses an identity decoder.
- Ignores `model.latent_dim` and `model.decoder_hidden_dims`.

Best when you want the simplest NODE baseline.

### `name: node2`

`NODE2` is a latent-space controlled Graph Neural ODE.

It encodes the history into a latent state, evolves that latent state, then decodes to physical state. In upgrade-v1, the default NODE2 config also enables residual decoding, history-conditioned latent dynamics, temporal Transformer history encoding, and normalized relative-time input:

```text
history -> latent -> ODE conditioned by forcing/history/time -> decoder + last_state -> future state
```

Important behavior:

- Uses `model.latent_dim`.
- Uses `model.decoder_hidden_dims`.
- Uses `model.decoder_activation`.
- Uses `model.dropout`.
- Uses `model.history_encoder`.
- Uses `model.continuous`.
- Uses `solver.ode_method`.
- Uses `model.use_residual_decoder`.
- Uses `model.use_history_in_ode`.
- Uses `model.use_relative_time`.
- Uses `model.relative_time_mode`.

Best when direct state-space dynamics are too restrictive.

### `name: ncde1`

`NCDE1` is a latent-space controlled Graph Neural CDE.

It treats future forcing plus time as a continuous control path:

```text
history -> latent -> CDE controlled by forcing/time -> decoder -> future state
```

Important behavior:

- Uses `model.latent_dim`.
- Uses `model.decoder_hidden_dims`.
- Uses `model.decoder_activation`.
- Uses `model.dropout`.
- Uses `model.history_encoder`.
- Uses `model.continuous`.
- Uses `solver.cde_method`.

Best when you want CDE-style controlled dynamics rather than ODE dynamics with forcing interpolation.

## Model Size Settings

These live directly under `model:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `latent_dim` | positive integer | Latent dimension for `node2` and `ncde1`. Larger values increase capacity and memory. Ignored by `node1`. |
| `dropout` | float from `0.0` to `<1.0` | Dropout used in `node2`/`ncde1` init MLP and decoder. |
| `decoder_hidden_dims` | list of integers | Hidden layers for `node2`/`ncde1` decoder. `[]` means a direct linear decoder. Ignored by `node1`. |
| `decoder_activation` | `relu`, `gelu`, `tanh`, `softplus`, `silu` | Activation used in `node2`/`ncde1` init MLP and decoder. |
| `decoder_chunk_size` | positive integer or `null` | Optional row chunk size for the latent decoder in `node2`/`ncde1`. Default `262144`; `null` disables chunking. |
| `use_residual_decoder` | `true` or `false` | NODE2-only. If `true`, decoder output is treated as a delta and added to the last observed normalized state. |
| `use_history_in_ode` | `true` or `false` | NODE2-only. If `true`, the latent ODE vector field receives `hist_context` at every function evaluation. |
| `use_relative_time` | `true` or `false` | NODE2-only. If `true`, the latent ODE vector field receives a normalized scalar relative time. |
| `relative_time_mode` | `normalized` | NODE2-only. Normalizes solver time by the final requested rollout time, giving a scalar in `[0, 1]`. |

Typical choices:

```yaml
model:
  latent_dim: 64
  decoder_hidden_dims: [64, 64]
```

or a larger model:

```yaml
model:
  latent_dim: 128
  decoder_hidden_dims: [128, 128]
```

## History Encoder Settings

These live under `model.history_encoder:`.

The history encoder is shared by all three models.

| Key | Supported values | What it does |
| --- | --- | --- |
| `static_hidden_dims` | list of integers | MLP hidden layers for static node features before graph/time encoding. |
| `static_embed_dim` | positive integer | Size of the static node embedding. |
| `gnn_hidden_dim` | positive integer | Hidden size inside the per-history-step GNN. |
| `gnn_num_layers` | integer `>= 1` | Number of message-passing layers in the history GNN. |
| `gnn_type` | `sage`, `gcn`, `graphconv` | Graph layer type. |
| `gnn_activation` | `relu`, `gelu`, `tanh`, `softplus`, `silu` | Activation used in the static MLP and history GNN. |
| `lstm_hidden_dim` | positive integer | Hidden size of the node-wise LSTM over history steps. |
| `lstm_num_layers` | integer `>= 1` | Number of LSTM layers. |
| `history_encoder_type` | `transformer`, `lstm` | Selects the temporal summarizer after per-step GNN encoding. |
| `use_transformer_history` | `true` or `false` | Enables the temporal-only Transformer when `history_encoder_type: transformer`. |
| `history_transformer_num_layers` | integer `>= 1` | Number of temporal Transformer encoder layers. |
| `history_transformer_num_heads` | integer `>= 1` | Number of attention heads; must divide `static_embed_dim`. |
| `history_transformer_ff_dim` | positive integer | Feed-forward width inside each temporal Transformer layer. |
| `history_transformer_dropout` | float from `0.0` to `<1.0` | Dropout inside temporal Transformer layers. |
| `history_transformer_chunk_size` | positive integer or `null` | Optional node-batch chunk size for temporal Transformer history. Default `8192`; `null` disables chunking. |
| `history_transformer_force_math_sdp` | `true` or `false` | On CUDA, force PyTorch's math scaled-dot-product attention backend for the temporal Transformer. Default `true`. |
| `history_use_positional_encoding` | `true` or `false` | Adds sinusoidal history-step positions before temporal self-attention. |
| `history_context_pooling` | `last`, `mean` | Extracts `hist_context` from the last Transformer token or the temporal mean. |
| `dropout` | float from `0.0` to `<1.0` | Dropout in static MLP/history GNN. LSTM dropout applies only when `lstm_num_layers > 1`. |

How changes behave:

- Larger `static_embed_dim`, `gnn_hidden_dim`, `lstm_hidden_dim`, or Transformer feed-forward width increases capacity and memory.
- Larger `history_len` gives the temporal summarizer more timesteps. The Transformer attends only across history steps for each node independently; spatial interaction still comes from the per-step GNN.
- `history_transformer_chunk_size` chunks only the node batch dimension. It does not shorten `history_len` or change what each node can attend to across history.
- `decoder_chunk_size` chunks only the flattened node-horizon rows sent through the decoder MLP. It lowers peak CUDA memory without changing the decoded tensor contract.
- `sage` is the default and usually robust.
- `gcn` is simpler but may be less expressive.
- `graphconv` is another PyG message-passing option.

## Continuous Dynamics Settings

These live under `model.continuous:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `hidden_dim` | positive integer | Hidden size inside the continuous dynamics GNN. |
| `num_layers` | integer `>= 1` | Number of message-passing layers in the dynamics function. |
| `gnn_type` | `sage`, `gcn`, `graphconv` | Graph layer type for the dynamics function. |
| `activation` | `relu`, `gelu`, `tanh`, `softplus`, `silu` | Activation inside the dynamics function. |
| `dropout` | float from `0.0` to `<1.0` | Dropout inside the dynamics function. |

How changes behave:

- Larger `hidden_dim` and `num_layers` increase dynamics capacity and cost.
- For `node1`, the continuous block predicts `dstate/dt`.
- For `node2`, the continuous block predicts `dlatent/dt`.
- For `ncde1`, the continuous block maps latent state to a control response tensor.

## Solver Settings

These live under `solver:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `interpolation` | `linear`, `hermite_cubic_backward` | Interpolation for known future forcing/control paths. This choice is enforced by repo code. |
| `ode_method` | string passed to `torchdiffeq.odeint` | ODE solver for `node1` and `node2`. Default in the base config files is `midpoint`. |
| `cde_method` | string passed to `torchcde.cdeint` | CDE solver for `ncde1`. Default in `model_ncde1.yaml` is `rk4`. |
| `rtol` | positive float | Relative tolerance for adaptive solvers. |
| `atol` | positive float | Absolute tolerance for adaptive solvers. |
| `use_adjoint` | `true` or `false` | If `true`, uses adjoint-mode integration where supported. Saves memory, may be slower or less stable. |
| `ode_options` | dictionary or `null` | Extra options passed to `torchdiffeq.odeint`. |
| `cde_options` | dictionary or `null` | Extra options passed to `torchcde.cdeint`. |

Common ODE/CDE method strings are backend-library choices. The repo does not validate them directly; it passes the string through and the library raises an error if unsupported.

Common method strings to try:

- `midpoint`
- `rk4`
- `euler`
- `dopri5`

Recommended defaults in this repo:

```yaml
solver:
  ode_method: midpoint
  cde_method: rk4
  interpolation: linear
  use_adjoint: false
```

For `NODE2` and `NCDE1`, the model config switches interpolation to:

```yaml
solver:
  interpolation: hermite_cubic_backward
```

Notes:

- `linear` is simpler and cheaper.
- `hermite_cubic_backward` gives a smoother control path.
- `rtol` and `atol` mainly matter for adaptive solvers. Fixed-step methods may ignore them or use options differently depending on the backend library.
- Keep `use_adjoint: false` for normal DDP runs unless you are explicitly testing adjoint behavior.

## Training Settings

These live under `training:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `epochs` | positive integer | Number of training epochs. |
| `batch_size` | positive integer | Training batch size. With large graphs, this is often `1`. |
| `lr` | positive float | AdamW learning rate. |
| `weight_decay` | float `>= 0` | AdamW weight decay. |
| `scheduler` | `cosine` or anything else | `cosine` enables cosine annealing. Any other value disables the scheduler. |
| `min_lr` | float `>= 0` | Minimum LR for cosine scheduler. Ignored if scheduler is disabled. |
| `grad_accum_steps` | positive integer | Accumulates gradients over this many mini-batches before optimizer step. |
| `max_grad_norm` | float | If `> 0`, clips gradient norm. If `0` or negative, disables clipping. |
| `train_horizon_mode` | `uniform_random`, `biased_long_horizon` | How to sample the effective rollout horizon during training. |
| `train_horizon_min` | positive integer | Minimum effective training horizon. |
| `train_horizon_max` | positive integer or `null` | Maximum effective training horizon. If `null`, uses `dataset.future_len`. |
| `train_horizon_curriculum.enabled` | `true` or `false` | Enables the epoch-wise curriculum cap before sampling `k_eff`. Default configs set this to `true`. |
| `train_horizon_curriculum.epochs` | integer `>= 0` | Number of early epochs used for curriculum ramp-up. Default: `120`. |
| `train_horizon_curriculum.warmup_fractions` | float list in `(0, 1]` | Stage fractions of the span from `train_horizon_min` to the target `train_horizon_max`. Default: `[0.40, 0.55, 0.70, 0.85]`. |
| `log_every` | integer | Currently read by the trainer but epoch logs are written every epoch. |
| `val_every` | positive integer | Runs validation every this many epochs. |

### Horizon Sampling

Training first resolves the epoch-local maximum horizon, samples an effective horizon `k_eff`, then truncates the future tensors before model forward:

- `force_future`
- `t_future`
- `y_future`
- future metadata such as `future_idx` and `future_time`

The model predicts only those first `k_eff` future steps for that training step, and the loss uses the full truncated prediction.
Fixed-window evaluation and full-rollout evaluation still use the full requested horizon.

`uniform_random`:

- samples `k_eff` uniformly from `[train_horizon_min, current_train_horizon_max]`
- treats short and long horizons equally

`biased_long_horizon`:

- samples from the same range
- gives larger horizons higher probability
- pushes the model to care more about longer rollouts

Important constraint:

```text
train_horizon_min <= train_horizon_max <= dataset.future_len
```

If `train_horizon_max: null`, it becomes `dataset.future_len`.

### Horizon Curriculum

The shared trainer supports a staircase curriculum for both ANUGA and ISSM. The target cap is still `training.train_horizon_max`; when it is `null`, the target is `dataset.future_len`. With the default curriculum:

```yaml
training:
  train_horizon_min: 24
  train_horizon_max: 120
  train_horizon_curriculum:
    enabled: true
    epochs: 120
    warmup_fractions: [0.40, 0.55, 0.70, 0.85]
```

the first 120 epochs use four equal-length stages:

```text
cap = train_horizon_min + fraction * (target_train_horizon_max - train_horizon_min)
```

rounded to the nearest integer and clamped to `[train_horizon_min, target_train_horizon_max]`. After the curriculum epochs, the cap stays at the target. For ISSM with `train_horizon_min: 24`, the target caps `40 / 60 / 80 / 100 / 120` produce approximate stage caps:

| Target max | Early stage caps | Final cap |
| ---: | --- | ---: |
| 40 | 30, 33, 35, 38 | 40 |
| 60 | 38, 44, 49, 55 | 60 |
| 80 | 46, 55, 63, 72 | 80 |
| 100 | 54, 66, 77, 89 | 100 |
| 120 | 62, 77, 91, 106 | 120 |

The CLI can override only the enable flag without adding another YAML file:

```bash
python scripts/train.py ... --horizon-curriculum off
python scripts/train.py ... --horizon-curriculum on
```

## Evaluation Settings

These live under `evaluation:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `batch_size` | positive integer | Batch size for validation/test fixed-window evaluation. |
| `num_workers` | integer `>= 0`, optional | Used by standalone `scripts/evaluate.py`. If absent, standalone eval uses `0`. Training-time eval uses `dataset.num_workers`. |
| `checkpoint_metric` | metric key string | Validation metric used to save `best.pt`. |
| `full_rollout_on_val` | `true` or `false` | If `true`, validation also runs full rollouts and checkpoint metric is read from rollout metrics. If `false`, checkpoint metric is read from fixed-window metrics. |
| `full_rollout_known_steps` | integer `>= history_len` or `null` | Optional training-time validation/test full-rollout start. If set, the first this many true timesteps are treated as known, but only the last `history_len` true steps are fed to the model. |

Default:

```yaml
evaluation:
  checkpoint_metric: whole_rollout_rmse
  full_rollout_on_val: true
  full_rollout_known_steps: null
```

That means the best checkpoint is selected by validation full-rollout physical RMSE.

If `full_rollout_known_steps` is `null`, full-rollout validation starts immediately after the configured history window. For example, `history_len: 4` predicts from step 5 onward.

If `full_rollout_known_steps: 60` and `history_len: 4`, validation/test full rollout treats steps 1-60 as known, feeds only steps 57-60 to the model, and predicts from step 61 onward.

### Checkpoint Metric Choices

If `full_rollout_on_val: true`, use full-rollout metric keys such as:

- `whole_rollout_rmse`
- `whole_rollout_mae`
- `whole_rollout_norm_rmse`
- `whole_rollout_norm_mae`
- `final_step_rmse`
- `final_step_mae`
- `final_step_norm_rmse`
- `final_step_norm_mae`
- channel-specific versions like `whole_rollout_rmse_ch0` or `final_step_norm_mae_ch2`

If `full_rollout_on_val: false`, use fixed-window metric keys such as:

- `rmse`
- `mae`
- `norm_rmse`
- `norm_mae`
- channel-specific versions like `rmse_ch0` or `norm_mae_ch2`

If the metric key does not exist, training will fail when it tries to save/check the best model.

## Distributed Settings

These live under `distributed:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `backend` | `nccl`, `gloo`, or `null` | Backend passed to PyTorch distributed initialization. `nccl` is normal for GPU DDP. `gloo` is safer for CPU. |
| `find_unused_parameters` | `true` or `false` | Passed to `DistributedDataParallel`. Keep `false` unless DDP reports unused parameter errors. |

DDP is enabled by launching with `torchrun` or `python -m torch.distributed.run`. The training `.sh` wrappers do this automatically.

`NPROC` is controlled by the shell script environment, not by YAML:

```bash
NPROC=2 ./train_issm_node1.sh
```

## AMP Settings

These live under `amp:`.

| Key | Supported values | What it does |
| --- | --- | --- |
| `mode` | `none`, `bf16`, `fp16` | Mixed precision mode on CUDA. |

Behavior:

- `none`: full precision, no autocast.
- `bf16`: bfloat16 autocast, no gradient scaler.
- `fp16`: float16 autocast with gradient scaler.

Use only these three strings. A typo may not fail early and can lead to unintended autocast behavior.

## Model Config Files

### `configs/model_node1.yaml`

```yaml
model:
  name: node1

solver:
  interpolation: linear
```

Use this for the simplest state-space NODE baseline.

### `configs/model_node2.yaml`

```yaml
model:
  name: node2
  latent_dim: 96
  decoder_hidden_dims: [96, 96]
  decoder_chunk_size: 262144
  use_residual_decoder: true
  use_history_in_ode: true
  use_relative_time: true
  relative_time_mode: normalized

solver:
  interpolation: hermite_cubic_backward
```

Use this for upgraded latent NODE dynamics. Add NODE2 ablation overlays after this file so their values override the upgraded defaults.

### NODE2 upgrade-v1 overlay configs

All files in this section live under `configs/NODE2_Upgrade1_Ablation/` and are overlays. Use them after `configs/model_node2.yaml`, not instead of it, so the run still selects `name: node2`, `latent_dim`, decoder sizes, and solver interpolation from the main NODE2 config.

Old baseline-like mode:

```yaml
# configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml
model:
  use_residual_decoder: false
  use_history_in_ode: false
  use_relative_time: false
  history_encoder:
    history_encoder_type: lstm
    use_transformer_history: false
    history_use_positional_encoding: false
```

Use this when you want the old NODE2 architecture as closely as possible:

```bash
--config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml \
--config configs/anuga.yaml \
--config configs/model_node2.yaml \
--config configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml
```

One-upgrade-only overlays:

| File | Effect |
| --- | --- |
| `configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml` | Enables only residual decoding; disables history-in-ODE, relative time, and Transformer history. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_history_in_ode_only.yaml` | Enables only history context inside the NODE2 ODE vector field; uses LSTM history and no residual/time upgrade. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_transformer_history_only.yaml` | Enables only Transformer history; disables residual decoder, history-in-ODE, and relative time. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_relative_time_only.yaml` | Enables only normalized relative time; uses LSTM history and no residual/history-in-ODE upgrade. |

Leave-one-out overlays from the full upgraded model:

| File | Effect |
| --- | --- |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_residual.yaml` | Full upgraded NODE2 except residual decoder is disabled. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_history_in_ode.yaml` | Full upgraded NODE2 except history context is not passed into the ODE vector field. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_transformer_history.yaml` | Full upgraded NODE2 except temporal history uses the old LSTM path. |
| `configs/NODE2_Upgrade1_Ablation/model_node2_no_relative_time.yaml` | Full upgraded NODE2 except relative time is not passed into the ODE vector field. |

### NODE2 upgrade-v2 structured vector field

`configs/model_node2_upgrade2.yaml` is an overlay used after
`configs/model_node2.yaml`. It changes only the NODE2 latent ODE vector field
from the v1 single graph network to `structured_v2`.

```yaml
model:
  node2_vector_field_type: structured_v2
  structured_dynamics:
    use_f_local: true
    use_f_spatial: true
    use_f_forcing: true
    use_f_coupling: true
    term_norm: rmsnorm
    fusion: softmax_gated
    gate_init: active_mean
```

Structured-v2 term switches:

| Key | Supported values | What it does |
| --- | --- | --- |
| `node2_vector_field_type` | `v1_base`, `structured_v2` | Selects the NODE2 latent vector-field implementation. Missing values default to `v1_base`. |
| `structured_dynamics.use_f_local` | `true` or `false` | Enables the node-local latent tendency branch. |
| `structured_dynamics.use_f_spatial` | `true` or `false` | Enables the graph-mediated spatial branch. |
| `structured_dynamics.use_f_forcing` | `true` or `false` | Enables the direct forcing branch. |
| `structured_dynamics.use_f_coupling` | `true` or `false` | Enables the branch that couples latent state, forcing, and spatial response. |

Structured-v2 scale-control settings:

| Key | Supported values | What it does |
| --- | --- | --- |
| `structured_dynamics.term_norm` | `none`, `layernorm`, `rmsnorm` | Normalizes each active branch over latent channels before fusion. `none` preserves the original raw branch outputs. |
| `structured_dynamics.fusion` | `sum`, `mean`, `direct_gated`, `softmax_gated` | Combines active branches after optional normalization. `sum` is the original raw-sum behavior; `mean` averages active branches; `direct_gated` learns one scalar gate per branch; `softmax_gated` learns one scalar logit per branch and softmaxes active logits. |
| `structured_dynamics.gate_init` | `active_mean` | Initializes direct gates to `1 / num_active_terms`. For `softmax_gated`, zero logits give the same equal active weights at initialization. |

For example, four active branches with `direct_gated` and
`gate_init: active_mean` start at `0.25` each. If coupling is disabled, the
three active branches start at `1/3` each. Do not combine active-mean gates
with an extra manual divide by the number of terms unless you intentionally
want to shrink the vector field again.

Shell-wrapper examples:

```bash
# Full upgraded model
RUN_NAME=anuga_node2_full_upgrade ./train_anuga_node2.sh

# Old baseline-like model
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml \
RUN_NAME=anuga_node2_old_baseline \
./train_anuga_node2.sh

# One upgrade at a time
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml RUN_NAME=anuga_node2_residual_only ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_history_in_ode_only.yaml RUN_NAME=anuga_node2_history_ode_only ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_transformer_history_only.yaml RUN_NAME=anuga_node2_transformer_only ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_relative_time_only.yaml RUN_NAME=anuga_node2_time_only ./train_anuga_node2.sh

# Leave-one-out ablations
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_no_residual.yaml RUN_NAME=anuga_node2_no_residual ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_no_history_in_ode.yaml RUN_NAME=anuga_node2_no_history_ode ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_no_transformer_history.yaml RUN_NAME=anuga_node2_no_transformer ./train_anuga_node2.sh
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_no_relative_time.yaml RUN_NAME=anuga_node2_no_time ./train_anuga_node2.sh
```

For queued jobs, pass `--inside-tmux` to the training wrapper so the scheduler job runs directly instead of launching a nested tmux session:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml \
RUN_NAME=anuga_node2_old_baseline \
./train_anuga_node2.sh --inside-tmux
```

### `configs/model_ncde1.yaml`

```yaml
model:
  name: ncde1
  latent_dim: 96
  decoder_hidden_dims: [96, 96]
  decoder_chunk_size: 262144

solver:
  interpolation: hermite_cubic_backward
  cde_method: rk4
```

Use this for latent CDE dynamics.

## Common Recipes

### Fast Debug Run

```yaml
training:
  epochs: 2
  batch_size: 1
  val_every: 1

dataset:
  sampled_windows:
    epoch_num_windows: 16
```

### Longer Training Windows

```yaml
dataset:
  history_len: 4
  future_len: 64

training:
  train_horizon_min: 8
  train_horizon_max: 64
```

This gives the encoder more history and trains on longer forecasts.

### Training-Horizon Ablation

For a clean long-rollout ablation, keep `dataset.future_len` fixed and vary only the target training cap:

```yaml
dataset:
  future_len: 120

training:
  train_horizon_min: 24
  train_horizon_max: 40  # or 60, 80, 100, 120
```

With the default curriculum enabled, each run ramps toward its own target cap for the first 120 epochs, then trains at that target for the remaining epochs.

### Fewer Windows Per Epoch

```yaml
dataset:
  sampled_windows:
    windows_per_scenario: 16
    epoch_num_windows: 1024
```

This is useful for long trajectories where using every possible window is too slow.

### Fixed-Window Checkpoint Selection

```yaml
evaluation:
  full_rollout_on_val: false
  checkpoint_metric: rmse
```

This selects `best.pt` by validation fixed-window physical RMSE instead of full-rollout RMSE.

### Normalized Metric Checkpoint Selection

```yaml
evaluation:
  full_rollout_on_val: true
  checkpoint_metric: whole_rollout_norm_rmse
```

This selects `best.pt` by normalized full-rollout RMSE.

### Delayed Full-Rollout Start

```yaml
evaluation:
  full_rollout_on_val: true
  checkpoint_metric: whole_rollout_rmse
  full_rollout_known_steps: 60
```

This selects `best.pt` using full-rollout validation that starts after 60 known timesteps. It is useful when deployment has an observed prefix, but the emulator should still receive only the last `history_len` true steps as input.

## Practical Gotchas

- Keep config merge order as base/history-scan config, then dataset config, then model config.
- For NODE2 ablations, put exactly one `configs/NODE2_Upgrade1_Ablation/model_node2_*.yaml` overlay after `configs/model_node2.yaml` unless you intentionally want multiple overlays to override each other from left to right.
- Use the same dataset and model config when evaluating a checkpoint.
- `NODE1` ignores `latent_dim` and decoder settings.
- `dataset.anuga`, `dataset.adcirc`, and `dataset.issm` are currently placeholders with no active options.
- `evaluation.num_workers` affects standalone fixed-window evaluation, but training-time validation/test loaders use `dataset.num_workers`.
- `training.log_every` is currently read but not used to skip epoch logs.
- `training.train_horizon_curriculum` changes only the maximum sampled training horizon per epoch; fixed-window and full-rollout evaluation still use their configured horizons.
- `full_rollout_on_val: true` is stricter but slower because it evaluates whole trajectories during validation.
- `full_rollout_known_steps` only changes training-time validation/test full-rollout start. Standalone full-rollout inference uses the CLI `--known-steps` flag.
- Larger `future_len`, `history_len`, `latent_dim`, `hidden_dim`, or GNN layer counts usually increase memory use.
- For large meshes, smaller `decoder_chunk_size` and `history_transformer_chunk_size` lower peak CUDA memory but can add extra kernel-launch overhead. Larger values can be faster if memory still fits.
