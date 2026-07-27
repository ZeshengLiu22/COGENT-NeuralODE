# Strict Codex Implementation Prompt — Unified Continuous-Time Graph Emulator Framework

You are Codex. Implement a **research-grade, modular, extensible PyTorch/PyG codebase** for a unified continuous-time graph emulator framework across three scientific simulation datasets:

- **ANUGA flood simulation**
- **ADCIRC storm surge simulation**
- **ISSM ice-sheet simulation**

This prompt is intentionally strict. **Do not improvise architecture changes, API changes, or repo layout changes unless required to make the code run.** When in doubt, follow this document literally.

---

## 0. Mission

Build a codebase that supports **controlled continuous-time graph dynamics** with **known future forcing** and a **unified dataset/model interface**.

The framework must support exactly these three baseline models for version 1:

1. `NODE1` = state-space controlled Graph Neural ODE
2. `NODE2` = latent-space controlled Graph Neural ODE
3. `NCDE1` = latent-space controlled Graph Neural CDE

The scientific objective is:

> Achieve **stable low error under long rollout**, rather than only strong one-step prediction.

Version 1 must prioritize:
- correctness
- clean abstractions
- reproducible rollout behavior
- DDP multi-GPU support
- mixed precision support
- future extensibility of the continuous block `f_theta`

---

## 1. Non-negotiable design rules

1. **Split by scenario/file first, then create windows.**
2. **Use a unified dataset API across all datasets.**
3. **Use a shared `HistoryEncoder` across all baselines.**
4. **Keep `state_hist` and `force_hist` separate in dataset code.**
5. **Keep the continuous block modular and swappable.**
6. **Use known future forcing during rollout.**
7. **Train with rollout MSE only in version 1.**
8. **Implement DDP and AMP in this round.**
9. **Implement Hermite cubic interpolation as an option in this round.**
10. **Do not add extra model families, attention variants, physics losses, or masking logic beyond placeholders.**
11. **For large meshes, treat 128/256 as effective batch size targets, not literal per-GPU PyG microbatch sizes.**
12. **Support gradient accumulation and sampled windows per epoch for long trajectories.**
13. **Sample `k_eff` once per global training step and synchronize it across DDP ranks.**

---

## 2. Tech stack and required libraries

Use:
- Python 3.10+
- PyTorch
- PyTorch Geometric
- `torchdiffeq`
- `torchcde`
- `PyYAML`
- `numpy`
- `scipy`

Optional but acceptable:
- `pandas`
- `tqdm`

Do **not** use DGL.

---

## 3. Repo layout (must follow this closely)

Create the repo with this structure:

```text
project_root/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── anuga.yaml
│   ├── adcirc.yaml
│   ├── issm.yaml
│   ├── model_node1.yaml
│   ├── model_node2.yaml
│   └── model_ncde1.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── anuga_dataset.py
│   ├── adcirc_dataset.py
│   ├── issm_dataset.py
│   ├── split_utils.py
│   ├── window_utils.py
│   └── normalization.py
│
├── models/
│   ├── __init__.py
│   ├── common/
│   │   ├── mlp.py
│   │   ├── gnn_blocks.py
│   │   ├── lstm_blocks.py
│   │   └── interpolation.py
│   │
│   ├── encoders/
│   │   └── history_encoder.py
│   │
│   ├── continuous/
│   │   ├── node_state_block.py
│   │   ├── node_latent_block.py
│   │   └── ncde_latent_block.py
│   │
│   ├── decoders/
│   │   ├── identity_decoder.py
│   │   └── mlp_decoder.py
│   │
│   ├── node1_model.py
│   ├── node2_model.py
│   └── ncde1_model.py
│
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── horizon_sampling.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_full_rollout.py
│
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── io.py
│   ├── logging_utils.py
│   └── shape_checks.py
│
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── predictions/
```

If you need to add a file to make the repo runnable, do so conservatively and document it.

---

## 4. Unified scientific problem setup

Each simulation file is one **fixed graph trajectory** consisting of:
- a fixed graph structure
- static node features
- dynamic forcing over time
- dynamic state over time

At node `i` and time `t`, conceptually:

$$
\mathbf{x}_i(t) = [\mathbf{s}_i, \mathbf{u}_i(t), \mathbf{y}_i(t)]
$$

where:
- `s_i`: static features
- `u_i(t)`: forcing
- `y_i(t)`: state

The models are **controlled**, not autonomous.

---

## 5. Dataset-specific provisional feature definitions

### 5.1 ANUGA

- `x_static`: x, y, elevation, friction
- `force`: rain rate
- `state`: water depth, x momentum, y momentum
- `scenario_id`: rainfall-case index
- optional future `scenario_param`: max rainfall intensity

### 5.2 ADCIRC

- `x_static`: spatial coordinates (x/y or lat/lon depending on preprocessing)
- `force`: `Vx`, `Vy`, pressure
- `state`: surge level
- `scenario_id`: year tag

### 5.3 ISSM

- `x_static`: x, y
- `force`: time-varying physical drivers such as melt rate, SMB, and related variables
- `state`: `Vx`, `Vy`, thickness
- `scenario_id`: melt-rate case ID
- optional future `scenario_param`: actual melt-rate scalar

Keep this flexible for future refinement, but implement the above as the initial design contract.

---

## 6. Dataset API (must be unified)

### 6.1 One sample = one temporal graph window

Use:
- history length `H`
- stored future length `K_max`

Each sample corresponds to a window ending at observed index `t` and contains:
- `state_hist = [t-H+1, ..., t]`
- `force_hist = [t-H+1, ..., t]`
- `force_future = [t+1, ..., t+K_max]`
- `y_future = [t+1, ..., t+K_max]`

### 6.2 Required fields in each PyG `Data` object

Each sample must include:
- `x_static` : `[N, F_static]`
- `state_hist` : `[N, H, F_state]`
- `force_hist` : `[N, H, F_force]`
- `force_future` : `[N, K_max, F_force]`
- `y_future` : `[N, K_max, F_state]`
- `edge_index` : `[2, E]`
- `scenario_id`
- `sim_id`
- `t_idx`

Optional placeholders:
- `scenario_param`
- `target_mask`
- `edge_attr`

### 6.3 PyG batching behavior

Design the code so that after PyG batching, tensors behave as:
- `x_static`: `[N_total, F_static]`
- `state_hist`: `[N_total, H, F_state]`
- `force_hist`: `[N_total, H, F_force]`
- `force_future`: `[N_total, K_max, F_force]`
- `y_future`: `[N_total, K_max, F_state]`

where `N_total` is the total number of nodes across the batched disconnected graphs.


### 6.4 Effective batch size versus microbatch size

For large full-graph continuous models, distinguish:
- per-GPU microbatch size
- number of GPUs
- gradient accumulation steps
- effective batch size

Use:

$$
B_{effective} = B_{micro} \times N_{gpu} \times N_{accum}
$$

For large ANUGA / ADCIRC / ISSM graphs, the implementation must assume that the per-GPU microbatch size is memory-limited, often 1 or 2. Achieve larger optimization-scale batch sizes such as 128 or 256 through DDP and gradient accumulation rather than by physically batching hundreds of full graph windows at once.

---

## 7. Split policy and window generation

### 7.1 Split policy

**Split by scenario/file first.**
Then create windows inside each split.

Default split semantics:
- ANUGA: by rainfall scenario / simulation file
- ADCIRC: by year / simulation file
- ISSM: by melt-rate scenario / simulation file

### 7.2 Window generation

For a trajectory of length `T`, valid window endpoints `t` must satisfy:
- history available through `t`
- future available through `t + K_max`

Support configurable:
- `history_len`
- `future_len`
- `stride`

### 7.3 Full-rollout evaluation

Support an evaluation mode that:
- observes the first `H` steps
- rolls out from `H+1` to the end using known future forcing
- computes errors over the whole trajectory, not only the endpoint

---

## 8. Normalization (must be implemented exactly like this)

Compute normalization statistics **from the training split only**.

Keep separate normalizers for:
- `x_static`
- `force`
- `state`

Use per-channel mean/std normalization with a standard-deviation floor:

$$
\tilde{x} = \frac{x - \mu}{\sigma_{safe}}
$$

where

$$
\sigma_{safe} =
\begin{cases}
\sigma, & \sigma \ge \sigma_{floor}, \\
1.0, & \sigma < \sigma_{floor}
\end{cases}
$$

Requirements:
- use a configurable standard-deviation floor, recommended default `1e-5`
- do **not** rely only on `sigma + eps` for near-constant channels
- apply the same train-set statistics to train/val/test
- train in normalized space
- inverse-transform predictions for physical-unit metrics

Implement a reusable normalizer class in `datasets/normalization.py`.

---

## 9. Shared `HistoryEncoder`

Implement exactly one shared `HistoryEncoder` used by all baselines.

### 9.1 Inputs
- `x_static`: `[N_total, F_static]`
- `state_hist`: `[N_total, H, F_state]`
- `force_hist`: `[N_total, H, F_force]`
- `edge_index`: `[2, E_total]`

### 9.2 Required architecture

1. Static embedding MLP:
$$
s_i = \mathrm{MLP}_{static}(x^{static}_i)
$$

2. For each history step `\tau`, build:
$$
r_i^\tau = [y_i^\tau, u_i^\tau, s_i]
$$

3. Pass `r^\tau` through a per-step GNN:
$$
h_i^\tau = \mathrm{GNN}_{step}(r^\tau, \mathcal{G})
$$

4. Stack over time and run a node-wise LSTM:
$$
c_i = \mathrm{LSTM}(h_i^{t-H+1}, \dots, h_i^t)
$$

### 9.3 Required outputs
Return a dict with:
- `static_embed`: `[N_total, D_enc]`
- `step_embeds`: `[N_total, H, D_enc]`
- `hist_context`: `[N_total, D_lstm]`
- `last_state`: `[N_total, F_state]`
- `last_force`: `[N_total, F_force]`

Do not omit `step_embeds`; keep it for future extensibility.

---

## 10. Continuous blocks versus full models

### Important conceptual distinction

The following are **continuous blocks** (the solver-called dynamics functions):
- `StateSpaceNODEFunc`
- `LatentNODEFunc`
- `LatentNCDEFunc`

They are **not wrappers around the whole model**.
They are only the `f_theta` parts called repeatedly during integration.

The following are **full assembled models**:
- `NODE1Model`
- `NODE2Model`
- `NCDE1Model`

Each full model owns the `HistoryEncoder`, the relevant continuous block, and the decoder if needed.

---

## 11. NODE1 = state-space controlled Graph NODE

### 11.1 Formula

Use the history encoder outputs and evolve the **physical state directly**:

$$
\frac{dy}{dt} = f_\theta\big(y(t), u(t), s, c, \mathcal{G}\big)
$$

where:
- `y(t)` = current physical state
- `u(t)` = interpolated future forcing
- `s` = static embedding
- `c` = history context from the LSTM

Initial condition:
$$
y(t_0) = y_{last}
$$

### 11.2 Required design

- use `HistoryEncoder`
- initial ODE state = `last_state`
- continuous block input must include:
  - current state
  - interpolated forcing
  - static embedding
  - history context
- decoder = identity

### 11.3 Inputs and outputs

Inputs:
- `x_static`
- `state_hist`
- `force_hist`
- `force_future`
- `edge_index`
- `t_future`

Output:
- `y_pred`: `[N_total, K, F_state]`

### 11.4 Integration

Use `torchdiffeq.odeint`.

Requirements:
- default large-mesh training solver method = `midpoint`
- support `rk4` as a higher-cost alternative
- support `dopri5` as an adaptive alternative
- support `euler` as a smoke-test option
- implement adjoint or checkpoint-style memory fallback as a configurable option, but not default unless needed

### 11.5 Activation guidance

Inside the continuous block, prefer smooth activations such as:
- `Softplus`
- `Tanh`

Do not use ReLU/LeakyReLU as the default inside the ODE dynamics block.

---

## 12. NODE2 = latent-space controlled Graph NODE

### 12.1 Formula

Build a latent initial state from history:

$$
z_0 = \mathrm{InitMLP}([c_{hist}, y_{last}, s_{static}])
$$

Then evolve latent dynamics:

$$
\frac{dz}{dt} = f_\theta\big(z(t), u(t), s, \mathcal{G}\big)
$$

Decode to physical state:

$$
\hat{y}(t) = \mathrm{DecMLP}(z(t))
$$

### 12.2 Required design

- use `HistoryEncoder`
- use an `InitMLP` to build `z0`
- continuous block input must include:
  - current latent state
  - interpolated forcing
  - static embedding
- history should enter via `z0`, not be re-injected every solver step by default
- decoder = small MLP

### 12.3 Inputs and outputs
Same external interface as NODE1.

### 12.4 Integration
Use `torchdiffeq.odeint` with the same solver configurability as NODE1.

---

## 13. NCDE1 = latent-space controlled Graph NCDE

### 13.1 Formula

Build latent initial state exactly as in NODE2:

$$
z_0 = \mathrm{InitMLP}([c_{hist}, y_{last}, s_{static}])
$$

Build a control path from future forcing and relative time:

$$
X_{ctrl}(t) = [u(t), t_{rel}]
$$

Evolve latent state with a graph NCDE:

$$
dz(t) = f_\theta(z(t), s, \mathcal{G}) \, dX_{ctrl}(t)
$$

Decode to physical state:

$$
\hat{y}(t) = \mathrm{DecMLP}(z(t))
$$

### 13.2 Required design

- use `HistoryEncoder`
- use `InitMLP` to build `z0`
- build control path from:
  - future forcing
  - relative rollout time
- continuous block should use:
  - latent state
  - static embedding
  - graph structure
- forcing must enter through the control path, not as an ordinary direct input
- decoder = small MLP

### 13.3 Interpolation requirements

Implement both:
- `linear`
- `hermite_cubic_backward`

Default NCDE control interpolation for large-mesh training:
- `linear`

Also support:
- `hermite_cubic_backward`

### 13.4 NCDE output-shape requirement

If the latent dimension is `D_latent` and the control dimension is `D_control`, then `LatentNCDEFunc` must produce an output tensor of shape:

$$
[N_{total}, D_{latent}, D_{control}]
$$

so that it matches the `torchcde` API contract.

### 13.5 Integration
Use `torchcde.cdeint`.

---

## 14. Interpolation module requirements

Implement interpolation utilities in `models/common/interpolation.py`.

### Required support

#### For NODE1 / NODE2 future forcing
- linear interpolation is required
- optional Hermite support is acceptable but not required as default

#### For NCDE1 control path
- linear interpolation
- Hermite cubic splines with backward differences

### Important notes
- Do not build a giant interpolation framework. Implement only what version 1 actually needs.
- Build interpolation coefficients once per batch / rollout, not inside every call to the continuous dynamics block.
- For large per-node-forcing runs, keep linear interpolation as the safe default and expose Hermite as a config option.

---

## 15. Training protocol

### 15.1 Version-1 loss
Use **rollout MSE only**.

No extra losses in version 1:
- no derivative loss
- no physics loss
- no horizon-weighted loss by default
- no mask-specific loss by default

### 15.2 Training horizon behavior
Store a full future block of length `K_max`, but during training sample an effective horizon `k_eff` before the model forward pass. Truncate future forcing, future times, targets, and matching future metadata to the first `k_eff` steps so the model only rolls out the supervised horizon for that training step.

Implement these horizon sampling modes:
1. `uniform_random` (**default**)
2. `biased_long_horizon` (**implemented but not default**)

#### Default recommendation
Use:
- `train_horizon_mode = uniform_random`
- `train_horizon_min = 2`
- `train_horizon_max = K_max`

### 15.3 DDP-safe horizon synchronization
`k_eff` must be sampled once per global training step and synchronized across DDP ranks before the forward pass. Do not sample `k_eff` independently inside `__getitem__` and do not let different ranks use different effective rollout lengths.

### 15.4 Teacher forcing
Do **not** use teacher forcing during future rollout.

### 15.5 Batch semantics
One batch is a batch of graph windows.

Use PyG batching naturally. Do not invent a custom graph batching system unless absolutely necessary.

### 15.6 Effective batch size and gradient accumulation
Support:
- memory-limited per-GPU microbatch size, often 1 or 2 for large meshes
- gradient accumulation
- effective batch size targets such as 128 or 256

The implementation must make it easy to achieve large effective batch sizes without requiring enormous literal PyG microbatches.

### 15.7 Sampled windows per epoch for long trajectories
Do not assume that every epoch must enumerate every possible valid graph window. Implement support for sampled-window training, for example:
- configurable windows-per-scenario per epoch
- or a configurable global epoch budget in number of graph windows

This is required to keep runtime manageable when simulation files are very long.

---

## 16. Evaluation protocol

Implement two evaluation modes.

### 16.1 Fixed-window evaluation
For validation and test:
- use the same window structure
- evaluate over the full future block
- report metrics in physical units after inverse normalization

### 16.2 Full-rollout evaluation
Required behavior:
- observe the first `H` steps
- rollout from `H+1` to the trajectory end using known future forcing
- compute metrics over the whole trajectory

### 16.3 Metrics to report
At minimum:
- RMSE in physical units
- MAE in physical units

For full-rollout evaluation, report:
- whole-rollout RMSE
- whole-rollout MAE
- final-step RMSE
- horizon-wise RMSE curve if practical

### 16.4 Checkpoint selection
Use validation RMSE in physical units as the default checkpoint metric.

---

## 17. DDP and mixed precision (required in this round)

### 17.1 DDP requirements
Use **PyTorch DDP via `torchrun`**.

Requirements:
- one process per GPU
- use `DistributedSampler`
- save checkpoints only on rank 0
- aggregate validation/test metrics across ranks correctly
- synchronize `k_eff` across ranks each training step

Do not use `DataParallel`.

### 17.2 Effective batch size requirements
The code must support the distinction between:
- per-GPU microbatch size
- number of GPUs
- gradient accumulation steps
- effective batch size

For large meshes, assume that the microbatch size may need to be as small as 1 or 2 per GPU. The implementation must support effective batch size targets such as 128 or 256 through DDP and gradient accumulation rather than by requiring giant literal PyG batches.

### 17.3 AMP requirements
Implement configurable AMP support with:
- `none`
- `bf16`
- `fp16`

Recommended safe version-1 behavior:
- keep model params and solver state in fp32 by default
- use autocast for neural network forward computations where safe
- use `GradScaler` only for fp16

Be conservative around continuous-time integration logic if needed for stability.

### 17.4 Memory fallback requirements
Implement at least one memory fallback for large-mesh training, such as:
- adjoint ODE backpropagation
- or checkpoint-style recomputation

This fallback should be configurable rather than mandatory as the default.

---

## 18. Required classes and responsibilities

### Dataset side
- `BaseTemporalGraphDataset`
- `ANUGADataset`
- `ADCIRCDataset`
- `ISSMDataset`
- `FeatureNormalizer`

### Model side
- `HistoryEncoder`
- `NODE1Model`
- `NODE2Model`
- `NCDE1Model`

### Continuous blocks
- `StateSpaceNODEFunc`
- `LatentNODEFunc`
- `LatentNCDEFunc`

### Training side
- `Trainer`
- `Evaluator`

Keep responsibilities clean and avoid mixing dataset logic into model classes.

---

## 19. Config requirements

Implement YAML-based config loading.

At minimum support config sections for:
- dataset
- normalization
- model
- training
- evaluation
- distributed/amp
- solver/interpolation

The code must support overriding defaults cleanly.

---

## 20. README requirements

The README must include:
- short project description
- supported datasets
- supported baselines
- repo layout overview
- how to train on a single GPU
- how to train with DDP using `torchrun`
- how to run evaluation
- how to run full-rollout evaluation

---

## 21. Implementation quality requirements

The code must be:
- readable
- modular
- type-hinted where reasonable
- documented with docstrings
- shape-aware
- explicit about tensor dimensions in critical modules

Add shape assertions in critical places.

Implement reasonable logging and error messages.

---

## 22. Acceptance criteria (must satisfy all)

The implementation is only complete if all of the following are true:

1. All three dataset classes exist and return a unified PyG `Data` format.
2. `HistoryEncoder` is shared and implemented exactly once.
3. `NODE1Model`, `NODE2Model`, and `NCDE1Model` all run end-to-end.
4. `torchdiffeq.odeint` is used for NODE1/NODE2.
5. `torchcde.cdeint` is used for NCDE1.
6. Linear interpolation is implemented for NODE forcing.
7. Linear and Hermite cubic backward interpolation are implemented for NCDE control.
8. DDP training is supported with `torchrun`.
9. AMP is supported with `none` / `bf16` / `fp16` modes.
10. Full-rollout evaluation is implemented.
11. Metrics are reported in physical units after inverse normalization.
12. Repo layout matches this prompt closely.
13. Version-1 loss is rollout MSE only.
14. Horizon sampling supports `uniform_random` and `biased_long_horizon`.
15. Code is organized so the continuous block can later be swapped with a topological neural network.
16. The code supports gradient accumulation and effective-batch-size-aware training for large meshes.
17. `k_eff` is synchronized across DDP ranks each step.
18. Normalization uses a standard-deviation floor for near-constant channels.
19. `LatentNCDEFunc` satisfies the `torchcde` output-shape contract `[N_total, D_latent, D_control]`.
20. Sampled-window training for long trajectories is supported.

---

## 23. Deliverables

You must produce:

1. the full codebase
2. configs
3. README
4. runnable training and evaluation scripts
5. minimal synthetic smoke-test utilities if needed to verify the pipeline

If synthetic smoke tests are added, keep them lightweight and clearly separated from the main scientific code.

---

## 24. What not to do

Do **not**:
- redesign the dataset API
- merge `state_hist` and `force_hist` in dataset code
- add extra baselines not requested
- add extra loss terms in version 1
- switch from PyG to DGL
- skip DDP or AMP
- omit Hermite support for NCDE
- replace the shared history encoder with three unrelated encoders
- silently deviate from the formulas in this prompt

---

## 25. Final instruction

Implement this as a **clean, research-ready version-1 codebase**.

When a choice is ambiguous, prefer:
- simpler code
- stronger modularity
- clearer tensor contracts
- easier future upgrades

But do not weaken any required functionality listed above.
