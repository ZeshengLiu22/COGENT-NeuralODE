# Continuous-Time Graph Emulator Master Handbook

> **Historical design reference.** This handbook includes proposed or removed
> NODE1/NCDE1 paths and earlier normalization, training, and evaluation rules.
> Those sections do not describe the current runtime. The active model is NODE2;
> use the [foundation repair report](foundation_correctness_repair.md),
> [current shell handbook](../handbook.md), [Upgrade V1](upgrade_v1.md), and
> [temporal-consistency guide](temporal_consistency.md) for current behavior.

**Project scope.** This master handbook consolidates the design decisions for a unified continuous-time graph learning framework spanning three scientific emulator datasets:

- **ANUGA flood simulation**
- **ADCIRC storm surge simulation**
- **ISSM ice-sheet simulation**

The central goal is not just strong one-step prediction, but **stable, low-error long rollout under known future forcing**, avoiding the error accumulation that often appears in standard autoregressive emulators.

This file is intended to serve both as:

- a **conceptual design handbook**,
- a **practical implementation specification**,
- and a **future reference manual** for extending the framework.

The framework is designed so that:

- all datasets share a **common input/output format**,
- only the **dataset adapters** need to change across domains,
- baseline models remain unified,
- and the core continuous dynamics block can later be upgraded to stronger spatial operators such as a **topological neural network**.

---

## 1. Project Goal and Design Philosophy

### 1.1 Main objective

We want a single framework that can support multiple scientific simulation datasets while preserving:

- a **unified model interface**,
- a **unified dataset interface**,
- and enough modularity to support future upgrades.

The central problem is:

> Traditional numerical simulators produce trajectories of graph-structured states under known external forcing. We want to learn continuous-time graph-based emulators that can roll forward accurately for long horizons without severe error accumulation.

### 1.1A How to use this handbook

You can read this file in three ways:

1. **As a project overview** for the scientific motivation and the common problem formulation.
2. **As an implementation handbook** for the dataset API, model contracts, training protocol, and code layout.
3. **As a research roadmap** for future upgrades such as stronger continuous blocks, topology-aware dynamics, richer losses, and improved rollout strategies.

### 1.1B Core problem formulation

Each simulation file is treated as a **trajectory on a fixed graph**.

For each scenario/run, we have:

- a fixed graph $\mathcal G = (\mathcal V, \mathcal E)$,
- static node features,
- time-varying forcing features,
- and time-varying state variables.

At node $i$ and time $t$, the node-wise information can be viewed as:

$$
\mathbf{x}_i(t) = [\mathbf{s}_i, \mathbf{u}_i(t), \mathbf{y}_i(t)],
$$

where:

- $\mathbf{s}_i$: static features,
- $\mathbf{u}_i(t)$: forcing,
- $\mathbf{y}_i(t)$: state.

The learning problem is:

> Given recent state history, recent forcing history, and known future forcing, predict the future state rollout on the same graph.

### 1.2 Why continuous-time models

Prior experiments suggest:

- **single-step** prediction can be very accurate,
- but **autoregressive rollout** can accumulate error,
- especially when the learned model is not explicitly designed for long-range continuous evolution.

This motivates continuous-time models such as:

- **Neural ODEs (NODEs)**
- **Neural Controlled Differential Equations (NCDEs)**

### 1.3 Autonomous vs controlled dynamics

For this project, the natural setting is **controlled dynamics**, not autonomous dynamics.

#### Autonomous ODE

An autonomous ODE evolves according to the current state alone:

$$
\frac{d z(t)}{dt} = f_\theta(z(t)).
$$

#### Controlled / non-autonomous ODE

A controlled ODE evolves according to the current state **and** external forcing:

$$
\frac{d z(t)}{dt} = f_\theta(z(t), u(t), s, \mathcal{G}),
$$

where:

- $z(t)$: evolving state or latent state,
- $u(t)$: forcing at time $t$,
- $s$: static node information,
- $\mathcal{G}$: graph structure.

This is the right paradigm for:

- rainfall-driven flooding,
- wind/pressure-driven surge,
- melt/SMB-driven ice-sheet evolution.

### 1.4 Design philosophy

We adopt the following principles:

1. **Keep dataset format fixed and simple.**
2. **Put flexibility in config and model modules, not in ad hoc dataset redesigns.**
3. **Use shared infrastructure across datasets.**
4. **Keep the first implementation simple but not too simple.**
5. **Design all continuous blocks to be swappable later**, for example replacing a simple GNN with a topological neural network.

---

## 2. Datasets and Provisional Feature Definitions

The current definitions are provisional and may be refined later, but they are sufficient for implementation round 1.

### 2.1 ANUGA

#### Static node features `x_static`

- x coordinate
- y coordinate
- elevation
- friction

#### Dynamic forcing `force`

- rain rate

#### Dynamic state `state`

- water depth
- x momentum
- y momentum

#### Scenario metadata

- `scenario_id`: rainfall-case index
- optional future `scenario_param`: max rainfall intensity

### 2.2 ADCIRC

#### Static node features `x_static`

- spatial coordinates (x/y or lat/lon, depending on preprocessing convention)

#### Dynamic forcing `force`

- wind velocity $V_x$
- wind velocity $V_y$
- atmospheric pressure

#### Dynamic state `state`

- surge level

#### Scenario metadata

- `scenario_id`: year tag

### 2.3 ISSM

#### Static node features `x_static`

- x coordinate
- y coordinate

#### Dynamic forcing `force`

- other time-varying physical drivers such as melt rate, surface mass balance, and related variables

#### Dynamic state `state`

- $V_x$
- $V_y$
- thickness

#### Scenario metadata

- `scenario_id`: melt-rate case ID
- optional future `scenario_param`: actual melt rate scalar

### 2.4 General abstraction across all three

Each simulation file is treated as:

- one **fixed graph**,
- with **static node features**,
- **dynamic forcing trajectory**,
- and **dynamic state trajectory**.

At time $t$, the node-wise information can be viewed as:

$$
\mathbf{x}_i(t) = [\mathbf{s}_i, \mathbf{u}_i(t), \mathbf{y}_i(t)],
$$

where:

- $\mathbf{s}_i$: static features,
- $\mathbf{u}_i(t)$: forcing,
- $\mathbf{y}_i(t)$: state.

---

## 3. Unified Dataset API

### 3.1 One sample = one temporal graph window

Each sample is a temporal window extracted from one simulation trajectory.

Given:

- history length: $H$
- stored future length: $K_{max}$

one sample contains:

- `state_hist`
- `force_hist`
- `force_future`
- `y_future`
- static features and graph structure

### 3.2 Common sample dictionary / `Data` fields

For each sample, the common fields are:

```python
x_static      # [N, F_static]
state_hist    # [N, H, F_state]
force_hist    # [N, H, F_force]
force_future  # [N, K_max, F_force]
y_future      # [N, K_max, F_state]
edge_index    # [2, E]
scenario_id   # metadata only
```

Optional future-compatible fields:

```python
edge_attr      # [E, F_edge] or None
scenario_param # optional physical scenario scalar or vector
target_mask    # optional, not used in version 1
```

### 3.3 Why `state_hist` and `force_hist` remain separate

We keep history tensors separate in the dataset:

- `state_hist`
- `force_hist`

and combine them only inside the model if needed.

This is cleaner because:

- it preserves the physical distinction,
- it gives more flexibility to model design,
- and it avoids baking one particular modeling assumption into the dataset.

---

## 4. Scenario Splits, Windowing, and Sampling

### 4.1 Split policy

The main rule is:

> **Split by scenario / simulation file first, then create windows.**

This avoids temporal leakage from overlapping windows of the same trajectory.

#### Dataset-specific split granularity

- **ANUGA**: split by rainfall scenario / simulation file
- **ADCIRC**: split by year / simulation file
- **ISSM**: split by melt-rate scenario / simulation file

### 4.2 Window definition

Let $t$ denote the last observed index in the history window.

Then:

$$
\text{state\_hist} = [y_{t-H+1}, \dots, y_t],
$$
$$
\text{force\_hist} = [u_{t-H+1}, \dots, u_t],
$$
$$
\text{force\_future} = [u_{t+1}, \dots, u_{t+K_{max}}],
$$
$$
\text{y\_future} = [y_{t+1}, \dots, y_{t+K_{max}}].
$$

### 4.3 Valid time indices

For a trajectory of total length $T$, a valid end-of-history index $t$ must satisfy:

$$
H - 1 \le t \le T - K_{max} - 1.
$$

Number of valid windows per simulation:

$$
T - H - K_{max} + 1.
$$

### 4.4 Stride

The dataset should support configurable stride:

- `stride = 1`: maximum overlap, more training samples
- `stride > 1`: less overlap, faster training, lower redundancy

### 4.5 Effective training horizon $k_{eff}$

We store a full future block of length $K_{max}$ in each dataset window, but during training we sample $k_{eff}$ before the model forward pass and truncate the future forcing, future times, targets, and matching metadata to the first $k_{eff}$ steps.

$$
1 \le k_{eff} \le K_{max}.
$$

This gives variable-horizon supervision and avoids rolling out unused future points during training. Evaluation still uses the full requested horizon.

### 4.6 Horizon sampling modes

#### Default

- `uniform_random`
- recommended default minimum horizon: `k_min = 2`

So:

$$
k_{eff} \sim \text{Uniform}(\{k_{min}, \dots, K_{max}\}).
$$

#### Also included in this implementation round

- `biased_long_horizon`

This is **implemented but not the default**.

#### Horizon curriculum

Horizon curriculum is supported as upgrade 1.1 through
`training.train_horizon_curriculum`. See `docs/upgrade_v1.1.md` for the
schedule, config keys, and ablation guidance.

#### Deferred for future upgrades

- `fixed`

---

## 5. Normalization Policy

### 5.1 Training-split-only rule

Normalization statistics must be computed **only from training scenarios**.

Never use validation or test scenarios to compute mean/std.

### 5.2 Separate normalizers

Keep separate per-channel normalizers for:

- static features
- forcing features
- state features

So we maintain:

- `static_norm`
- `force_norm`
- `state_norm`

### 5.3 Per-channel z-score normalization

For any channel $x$, use:

$$
\tilde{x} = \frac{x - \mu}{\sigma_{safe}},
$$

where $\sigma_{safe}$ is defined by a standard-deviation floor rule rather than only by adding a tiny epsilon:

$$
\sigma_{safe} =
\begin{cases}
\sigma, & \sigma \ge \sigma_{floor}, \\
1.0, & \sigma < \sigma_{floor}.
\end{cases}
$$

Recommended default:

- $\sigma_{floor} = 10^{-5}$
- still keep a tiny implementation epsilon internally if desired, but do **not** rely on $\sigma + \varepsilon$ alone for near-constant channels

This avoids pathological scaling when a channel is physically constant or nearly constant in the training set, which can happen for dry or inactive regions in scientific simulation data.

### 5.4 Aggregation dimensions

#### Static features

Compute mean/std across:

- training scenarios
- all nodes

#### Forcing features

Compute mean/std across:

- training scenarios
- all time steps
- all nodes

#### State features

Compute mean/std across:

- training scenarios
- all time steps
- all nodes

### 5.5 Metrics in physical space

Training can happen in normalized space, but reported metrics should be computed after inverse transformation to physical units.

---

## 6. PyG Data Objects, Batching, and DDP Considerations

### 6.1 One `Data` object per temporal window

Each sample returned by the dataset is one `torch_geometric.data.Data` object representing:

- one graph
- plus temporal tensors attached to its nodes

### 6.2 Where to build the `Data` object

Recommended pattern:

- build reusable trajectory-level information in dataset `__init__`
- construct `Data` objects in `__getitem__`

### 6.3 PyG batching

In PyG, batching concatenates multiple graphs into one disconnected graph.

If one sample has:

- $N$ nodes
- $E$ edges

then a batch of $B$ samples becomes approximately:

- $B N$ nodes
- $B E$ edges

PyG automatically:

- concatenates node features,
- offsets `edge_index`,
- builds a `batch` vector.

### 6.4 Batched shapes

After PyG batching, we work with:

- `x_static`: `[N_total, F_static]`
- `state_hist`: `[N_total, H, F_state]`
- `force_hist`: `[N_total, H, F_force]`
- `force_future`: `[N_total, K_max, F_force]`
- `y_future`: `[N_total, K_max, F_state]`
- `edge_index`: `[2, E_total]`
- `batch`: `[N_total]`

### 6.5 DDP compatibility

This batching scheme is naturally compatible with PyTorch DDP:

- each rank receives a subset of graph windows,
- each batch remains a standard PyG batched graph,
- the model interface does not change.


### 6.6 Effective batch size versus microbatch size

For large full-graph continuous models, it is crucial to distinguish:

- **microbatch size** = how many graph windows fit in one actual forward/backward pass on one GPU,
- **world size** = number of GPUs,
- **gradient accumulation steps** = how many microbatches are accumulated before one optimizer step,
- **effective batch size** = the optimization-scale batch size seen by the optimizer.

Use:

$$
B_{effective} = B_{micro} \times N_{gpu} \times N_{accum}.
$$

For large ANUGA / ADCIRC / ISSM graphs, the default assumption should be:

- $B_{micro}$ is **memory-limited**, often 1 or 2 per GPU,
- target effective batch size can still be 128 or 256 via DDP and gradient accumulation.

Do **not** assume that a literal PyG batch of 128 or 256 full graph windows is feasible when each graph window contains tens of thousands of nodes.

### 6.7 Sampled windows per epoch versus exhaustive enumeration

A long trajectory can yield a very large number of valid windows. For large full-graph continuous models, one training epoch does **not** need to exhaustively iterate over every possible window.

Instead, support a sampled-window training regime where each epoch uses:

- a configurable number of windows per scenario,
- or a configurable global epoch budget in number of graph windows,
- while still preserving scenario-level splits and randomization over start times and horizons.

This keeps runtime manageable for long trajectories while still exposing the model to diverse temporal contexts across epochs.

---

## 7. Shared Module Design

### 7.1 Main categories

#### Dataset side

- `BaseTemporalGraphDataset`
- `ANUGADataset`
- `ADCIRCDataset`
- `ISSMDataset`

#### Model side

- `HistoryEncoder`
- `NODE1Model`
- `NODE2Model`
- `NCDE1Model`

#### Continuous dynamics blocks

- `StateSpaceNODEFunc`
- `LatentNODEFunc`
- `LatentNCDEFunc`

### 7.2 Difference between full models and continuous blocks

The **continuous block** is **not** the whole model.

The continuous block is the internal module that the continuous-time solver repeatedly calls.

For example:

- `StateSpaceNODEFunc` = $f_\theta$ for NODE1
- `LatentNODEFunc` = $f_\theta$ for NODE2
- `LatentNCDEFunc` = the NCDE dynamics function

Meanwhile:

- `NODE1Model` = `HistoryEncoder + StateSpaceNODEFunc + solver + decoder`
- `NODE2Model` = `HistoryEncoder + init block + LatentNODEFunc + solver + decoder`
- `NCDE1Model` = `HistoryEncoder + init block + LatentNCDEFunc + CDE solver + decoder`

### 7.3 Why this modularity matters

This makes it easy to later replace the continuous block with:

- a stronger GNN,
- a topological neural network,
- a physics-aware message-passing block,
- or another continuous dynamics operator.

---

## 8. HistoryEncoder

### 8.1 Purpose

The HistoryEncoder summarizes the known past for each node.

It is shared by:

- NODE1
- NODE2
- NCDE1

### 8.2 Input contract

Given a PyG-batched graph window:

- `x_static`: `[N_total, F_static]`
- `state_hist`: `[N_total, H, F_state]`
- `force_hist`: `[N_total, H, F_force]`
- `edge_index`: `[2, E_total]`

### 8.3 Output contract

The HistoryEncoder returns:

- `static_embed`: `[N_total, D_enc]`
- `step_embeds`: `[N_total, H, D_enc]`
- `hist_context`: `[N_total, D_lstm]`
- `last_state`: `[N_total, F_state]`
- `last_force`: `[N_total, F_force]`

### 8.4 Architecture

#### Static embedding

$$
s_i = \mathrm{MLP}_{static}(x^{static}_i).
$$

#### Per-step graph encoding

For each history step $\tau$:

$$
r_i^{\tau} = [y_i^{\tau}, u_i^{\tau}, s_i],
$$
$$
h_i^{\tau} = \mathrm{GNN}_{step}(r^{\tau}, \mathcal{G}).
$$

#### Node-wise LSTM

For each node:

$$
c_i = \mathrm{LSTM}(h_i^{t-H+1}, \dots, h_i^t).
$$

The final hidden state is used as:

$$
\text{hist\_context}_i = c_i.
$$

### 8.5 Role in the whole pipeline

The HistoryEncoder does **not** solve any ODE or CDE. It only summarizes the known history.

---

## 9. NODE1: State-Space Controlled Graph Neural ODE

### 9.1 Purpose

NODE1 evolves the **physical state directly**.

It answers:

> If we continuously evolve the simulator state itself, can a controlled graph ODE achieve stable long rollout?

### 9.2 Model formula

Let $y(t)$ denote the physical state, $u(t)$ the future forcing, $s$ the static embedding, and $c$ the history context.

The continuous dynamics are:

$$
\frac{d y(t)}{dt} = f_\theta\big(y(t), u(t), s, c, \mathcal{G}\big).
$$

### 9.3 Initial condition

The initial ODE state is the last observed physical state:

$$
y(t_0) = y_{last}.
$$

### 9.4 Decoder

Identity decoder:

$$
\hat{y}(t) = y(t).
$$

### 9.5 Internal structure

1. Run `HistoryEncoder`
2. Extract:
   - `last_state`
   - `static_embed`
   - `hist_context`
3. Build future forcing interpolation
4. Solve the ODE from $t_0$ over future times
5. Return the predicted future physical trajectory

### 9.6 NODE1 forward summary

$$
\hat{Y}_{future} = \mathrm{ODEsolve}\big(f_\theta, y_{last}, u(t), s, c\big).
$$

### 9.7 Why NODE1 matters

NODE1 is the most interpretable baseline because the continuous state is the actual physical state.

---

## 10. NODE2: Latent-Space Controlled Graph Neural ODE

### 10.1 Purpose

NODE2 evolves a **latent state**, not the physical state directly.

It answers:

> Does compressing the history into a latent representation and evolving that latent state improve long-rollout stability?

### 10.2 Initial latent state

Run the HistoryEncoder and define:

$$
z_0 = \mathrm{InitMLP}\big([c_{hist}, y_{last}, s_{static}]\big).
$$

### 10.3 Latent continuous dynamics

$$
\frac{d z(t)}{dt} = f_\theta\big(z(t), u(t), s, \mathcal{G}\big).
$$

### 10.4 Decoder

A small per-node decoder MLP maps latent state back to physical state:

$$
\hat{y}(t) = \mathrm{DecMLP}(z(t)).
$$

### 10.5 NODE2 forward summary

$$
z_0 = \mathrm{InitMLP}\big([c_{hist}, y_{last}, s]\big),
$$
$$
Z_{future} = \mathrm{ODEsolve}\big(f_\theta, z_0, u(t), s\big),
$$
$$
\hat{Y}_{future} = \mathrm{DecMLP}(Z_{future}).
$$

### 10.6 Why NODE2 matters

NODE2 is often a stronger baseline than state-space evolution when the physical state is hard to evolve directly in a stable manner.

---

## 11. NCDE1: Latent-Space Controlled Graph Neural CDE

### 11.1 Purpose

NCDE1 uses a continuous control path rather than feeding forcing directly into an ODE vector field.

It answers:

> Compared with latent NODE, does control-driven latent continuous evolution give better long-rollout stability?

### 11.2 Initial latent state

Same initialization philosophy as NODE2:

$$
z_0 = \mathrm{InitMLP}\big([c_{hist}, y_{last}, s]\big).
$$

### 11.3 Control path

Define the control path from future forcing plus relative time:

$$
X_{ctrl}(t) = [u(t), t_{rel}].
$$

### 11.4 Latent NCDE dynamics

$$
d z(t) = f_\theta\big(z(t), s, \mathcal{G}\big)\, dX_{ctrl}(t).
$$

The NCDE function maps latent state to a response with respect to the control channels.

### 11.5 Decoder

As in NODE2:

$$
\hat{y}(t) = \mathrm{DecMLP}(z(t)).
$$

### 11.6 NCDE1 forward summary

$$
z_0 = \mathrm{InitMLP}\big([c_{hist}, y_{last}, s]\big),
$$
$$
Z_{future} = \mathrm{CDEsolve}\big(f_\theta, z_0, X_{ctrl}(t)\big),
$$
$$
\hat{Y}_{future} = \mathrm{DecMLP}(Z_{future}).
$$

### 11.7 NODE2 vs NCDE1

- **NODE2**: forcing enters as a direct input to the vector field
- **NCDE1**: forcing defines the control path itself

This is the key conceptual difference.

---

## 12. Interpolation and Integration Choices

### 12.1 Forcing interpolation for NODE1 and NODE2

For NODE models, future forcing must be available at intermediate continuous times during the ODE solve.

#### Default

- `linear`

This remains the default for version 1 because it is simple, stable, and relatively cheap for large meshes.

### 12.2 Control interpolation for NCDE1

For NCDE, interpolation defines the continuous control path.

#### Included in this implementation round

- `linear`
- `hermite_cubic_backward`

### 12.3 Practical interpolation policy for large meshes

Hermite cubic interpolation is included in this implementation round because it can be a stronger option for NCDE-style control paths. However, for very large meshes with per-node forcing, spline construction itself can become a nontrivial computational bottleneck.

Therefore:

- **implemented options**: `linear`, `hermite_cubic_backward`
- **safe large-mesh default**: `linear`
- **recommended upgrade option for smaller or better-optimized runs**: `hermite_cubic_backward`

Interpolation coefficients must be built **once per batch / rollout**, outside the inner continuous block, and must **not** be recomputed inside every call to the continuous dynamics function.

### 12.4 ODE integration backend

For NODE1 and NODE2:

- use `torchdiffeq.odeint`

#### Recommended training defaults

For large-mesh training runs, prefer a memory- and compute-conscious default such as:

- `midpoint` as the default training solver

Also support:

- `rk4` as a higher-cost alternative
- `dopri5` as an adaptive alternative
- `euler` as a smoke-test option

#### Adjoint / checkpoint fallback

Use ordinary odeint backpropagation for the active NODE2 implementation.

### 12.5 CDE integration backend

For NCDE1:

- use `torchcde.cdeint`

#### Default control interpolation choice

Recommended version-1 default for large-mesh NCDE1 training:

- `linear`

#### Also supported

- `hermite_cubic_backward`

### 12.6 NCDE output-shape requirement

For NCDE1, the continuous block must satisfy the `torchcde` API contract. If the latent dimension is $D_{latent}$ and the control dimension is $D_{ctrl}$, then the NCDE function must output a tensor shaped like:

$$
[N_{total}, D_{latent}, D_{ctrl}].
$$

This is required so that the latent response can be multiplied by the derivative of the control path.

### 12.7 Smooth activations in continuous blocks

Inside continuous blocks, prefer smoother activations such as:

- `Softplus`
- `Tanh`

rather than highly non-smooth choices, to make continuous optimization and integration more stable.

---

## 13. Training Protocol

### 13.1 Training goal

Training should optimize for stable multi-step rollout, not merely one-step prediction.

### 13.2 Training targets

All models predict future state rollout:

$$
\hat{Y}_{future} \in \mathbb{R}^{N \times K_{max} \times F_{state}}.
$$

### 13.3 Effective horizon

During training, the batch is truncated to the first $k_{eff}$ future steps before the model forward pass:

$$
\hat{Y}_{1:k_{eff}}, \quad Y_{1:k_{eff}}.
$$

The model therefore predicts only $k_{eff}$ rollout points for that training step, and all predicted points contribute to the loss.

### 13.4 DDP-safe horizon sampling

In distributed training, $k_{eff}$ must be sampled **once per global training step**, not independently inside each dataset sample and not independently per rank.

Recommended rule:

1. sample $k_{eff}$ on rank 0,
2. broadcast it to all ranks,
3. use the same $k_{eff}$ for the entire distributed step.

This guarantees shape consistency across ranks, prevents DDP synchronization failures, and keeps all ranks rolling out the same sampled horizon.

### 13.5 No teacher forcing during rollout

Once the rollout begins, the continuous model should rely on:

- encoded history,
- initial condition,
- and known future forcing,

but **not** on true future states.

### 13.6 Version-1 loss

Use rollout MSE only:

$$
\mathcal{L}_{rollout} = \frac{1}{N\, k_{eff}\, F_{state}} \sum_{i,t,c} \left(\hat{y}_{i,t,c} - y_{i,t,c}\right)^2.
$$

No additional losses in version 1.

### 13.7 Channel weighting

In version 1, keep the loss simple. Channel weights may be added later, but are not part of the default baseline.

### 13.8 Training horizon modes

#### Default

- `uniform_random`
- recommended default: `k_min = 2`

#### Also implemented

- `biased_long_horizon`

#### Curriculum

Horizon curriculum is implemented as upgrade 1.1. See
`docs/upgrade_v1.1.md`.

#### Deferred

- `fixed`

### 13.9 Effective batch size, microbatch size, and gradient accumulation

For large meshes, batch semantics must be memory-aware. Distinguish:

- **microbatch size** per GPU,
- **number of GPUs**,
- **gradient accumulation steps**,
- and **effective batch size** seen by the optimizer.

Use:

$$
B_{effective} = B_{micro} \times N_{gpu} \times N_{accum}.
$$

Recommended default mindset for large ANUGA / ADCIRC / ISSM graphs:

- $B_{micro}$ is memory-limited and often 1 or 2 per GPU,
- target effective batch size can still be 128 or 256 through DDP and gradient accumulation.

### 13.10 Sampled windows per epoch for long trajectories

When trajectories are long, exhaustively iterating over every valid graph window in every epoch may be prohibitively slow. Therefore, version 1 should support a sampled-window training regime where each epoch uses:

- a configurable number of windows per scenario,
- or a configurable global epoch budget in graph windows,
- while still randomizing start times and horizon sampling.

This keeps runtime manageable while maintaining good temporal diversity across epochs.

### 13.11 Practical training defaults for large meshes

For large-mesh runs, the most practical default is:

- small per-GPU microbatch,
- DDP,
- gradient accumulation,
- sampled windows per epoch,
- rollout MSE only,
- and known future forcing during rollout.


---

## 14. Evaluation Protocol

### 14.1 Two evaluation modes

#### A. Window-based evaluation

Use the same fixed-window format as training, but evaluate over the full future block.

This measures local multi-step forecasting quality.

#### B. Full-rollout evaluation

Given the first observed history, roll forward continuously to the end of the trajectory using known future forcing.

This directly tests long-horizon stability.

### 14.2 Full-rollout start

#### Default

Start rollout immediately after the history window:

- observe first $H$ steps,
- predict from $H+1$ to the end.

#### Delayed start with known prefix

Some workflows know a longer observed prefix than the model should consume directly. The evaluation protocol therefore separates:

- `history_len`: number of recent true state/forcing steps fed into the emulator,
- `full_rollout_known_steps`: number of true trajectory steps treated as known before rollout begins.

If `full_rollout_known_steps` is `null`, the default start above is used. If `history_len = 4` and `full_rollout_known_steps = 60`, then:

- true steps 1-60 are available as the known prefix,
- only true steps 57-60 are passed to the model as history,
- rollout prediction starts at step 61 and continues to the end.

This matches the standalone full-rollout CLI behavior, where `--known-steps K` is converted to `start_t = K - 1` and `get_rollout_data` slices only the last `history_len` true steps as model input.

### 14.3 Why full-rollout evaluation is required

This project is specifically motivated by long-range autoregressive drift. Therefore, evaluating only short windows is insufficient.

### 14.4 What to measure in full-rollout evaluation

At minimum:

- whole-rollout RMSE
- whole-rollout MAE
- final-step RMSE
- horizon-wise RMSE curve

Optional later:

- first-half vs second-half rollout error
- peak-event metrics
- dataset-specific diagnostics

### 14.5 Model selection

Recommended checkpointing criterion:

- validation rollout RMSE in physical units

This aligns model selection with the real forecasting goal.

---

## 15. Losses and Metrics

### 15.1 Training loss (version 1)

Only:

- rollout MSE

### 15.2 Deferred future losses

Possible future upgrades:

- horizon-weighted loss
- derivative consistency loss
- physics-informed loss
- wet-region weighting
- channel-balancing
- long-horizon emphasis terms

These are intentionally excluded from version 1 to keep the baseline clean.

### 15.3 Reported metrics

After inverse normalization, report metrics in physical units.

#### ANUGA

- RMSE/MAE for water depth
- RMSE/MAE for x momentum
- RMSE/MAE for y momentum

#### ADCIRC

- RMSE/MAE for surge

#### ISSM

- RMSE/MAE for $V_x$
- RMSE/MAE for $V_y$
- RMSE/MAE for thickness

---

## 16. Codebase Structure

Recommended layout:

```text
project_root/
├── README.md
├── requirements.txt
├── configs/
│   ├── base_ANUGA.yaml
│   ├── base_ISSM.yaml
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

### 16.0 Config side

Config bundles are merged from left to right. New runs should use a dataset-specific base config first, then the dataset locator/split config, then the model config:

```bash
# ANUGA
--config configs/base_ANUGA.yaml --config configs/anuga.yaml --config configs/model_node2.yaml

# ISSM
--config configs/base_ISSM.yaml --config configs/issm.yaml --config configs/model_node2.yaml
```

`base_ANUGA.yaml` and `base_ISSM.yaml` hold dataset-oriented defaults for history length, forecast window length, sampled-window speed controls, training cadence, evaluation checkpointing, DDP, and AMP. The dataset files such as `anuga.yaml` and `issm.yaml` should stay focused on data paths, file patterns, and split rules. Model files should stay focused on architecture selection and architecture-specific overrides.

`base.yaml` is retained only as a legacy/shared base for compatibility and for datasets that do not yet have a dedicated base file.

### 16.1 Dataset side

- `BaseTemporalGraphDataset`: common split/window/normalization logic
- dataset subclasses: raw-format adapters

### 16.2 Model side

- `HistoryEncoder`: shared front end
- `NODE1Model`, `NODE2Model`, `NCDE1Model`: full assembled models

### 16.3 Continuous side

- `StateSpaceNODEFunc`: derivative function for NODE1
- `LatentNODEFunc`: derivative function for NODE2
- `LatentNCDEFunc`: CDE dynamics function for NCDE1

### 16.4 Training side

- `losses.py`: rollout MSE and future loss extensions
- `horizon_sampling.py`: uniform + biased-long modes
- `trainer.py`: train/validation loop, checkpoint selection, and optional delayed full-rollout validation via `evaluation.full_rollout_known_steps`
- `evaluator.py`: fixed-window metrics and full-rollout metrics, including explicit `start_t` support for delayed rollout starts
- `trainer.py`, `evaluator.py`, `metrics.py`

---

## 17. Speed, Multi-GPU, and Mixed Precision

### 17.1 DDP

Include DDP in this implementation round.

Recommended pattern:

- use `torchrun`
- one process per GPU
- use `DistributedSampler`
- save checkpoints only on rank 0

### 17.2 Why DDP is compatible

Because each sample is just one graph window, DDP can distribute windows across ranks naturally.

### 17.3 Effective batch size policy

For large full-graph continuous models, the optimization target batch size should usually be understood as an **effective batch size**, not a literal PyG microbatch of hundreds of giant graph windows.

Recommended large-mesh policy:

- per-GPU microbatch size is memory-limited, often 1 or 2,
- use gradient accumulation to reach an effective batch size target such as 128 or 256,
- do not assume that physically collating 128 or 256 full graph windows into one PyG batch is feasible.

### 17.4 Runtime control for long trajectories

Long simulation files can make exhaustive window training very slow even when memory fits. Therefore, support:

- sampled windows per epoch,
- configurable epoch budget in windows,
- configurable windows-per-scenario budget,
- and gradient accumulation.

These are part of the practical runtime strategy for version 1.

### 17.5 Mixed precision

Include AMP support in this implementation round.

Recommended config options:

- `none`
- `bf16`
- `fp16`

#### Safer default

- neural networks under autocast
- solver state, time tensors, and interpolation in fp32

This balances speed and stability.

### 17.6 GradScaler

- use when `fp16`
- typically unnecessary for `bf16`

### 17.7 Why keep solver logic in fp32 initially

Continuous-time integration can be more numerically sensitive than standard feedforward inference, so keeping core solver state in fp32 is a prudent version-1 decision.

### 17.8 Memory fallback options

For large-mesh continuous models, implement a memory fallback such as:

- ordinary ODE backpropagation,
- checkpoint-style recomputation where appropriate,
- chunked independent decoder and temporal-history operations,
- and reduced-cost solvers for training.

These should be available as configuration options even if they are not the default.


---

## 18. Future Upgrades and Research Directions

This project is intentionally designed to support future upgrades.

### 18.1 Stronger continuous blocks

The continuous block is the most natural place to replace a simple GNN with:

- a topological neural network,
- a physics-aware graph operator,
- higher-order message passing,
- or hybrid geometric/topological dynamics.

### 18.2 Stronger history encoders

Possible future upgrades:

- Transformer over history
- temporal attention
- continuous-time history encoding
- graph-temporal attention

### 18.3 More advanced interpolation

Possible future additions:

- more interpolation options for NODE forcing
- richer CDE control interpolation variants
- learned interpolation if needed

### 18.4 Better long-horizon training

Possible future additions:

- biased-long-horizon default
- horizon-weighted losses
- rollout consistency regularization

### 18.5 Masks and scenario conditioning

Future additions:

- `target_mask`
- wet/dry auxiliary masks for ANUGA
- explicit `scenario_param` conditioning
- conditional decoders or FiLM-style conditioning

---

## 19. Implementation Checklist for This Round

### Confirmed

- unified dataset API
- split by scenario first
- fixed `H`, fixed stored `K_max`
- variable effective training horizon $k_{eff}$
- default horizon mode: `uniform_random`
- optional implemented mode: `biased_long_horizon`
- version-1 loss: rollout MSE only
- HistoryEncoder shared across all baselines
- NODE1, NODE2, NCDE1 as the baseline suite
- `torchdiffeq.odeint` for NODE1/NODE2 with memory-aware solver configurability
- `torchcde.cdeint` for NCDE1
- include Hermite cubic interpolation option in this round, but keep linear available as the safe large-mesh default
- include DDP, mixed precision, gradient accumulation, and effective-batch-size-aware training in this round
- synchronize `k_{eff}` across DDP ranks each step
- implement a standard-deviation floor in normalization
- enforce the NCDE output-shape contract `[N_total, D_{latent}, D_{ctrl}]`

### Baseline suite for this implementation round

1. **NODE1**: state-space controlled Graph NODE
2. **NODE2**: latent-space controlled Graph NODE
3. **NCDE1**: latent-space controlled Graph NCDE

### Final philosophy for this round

Keep the implementation:

- **simple enough to run reliably**
- **modular enough to evolve**
- **fair enough for comparison across baselines**
- **robust enough for future replacement of the continuous block**

---

## Closing note

This handbook is the design reference for the current implementation round. It is intended to serve both as:

- a conceptual summary of the project,
- and a practical implementation specification for coding, debugging, and future extensions.
