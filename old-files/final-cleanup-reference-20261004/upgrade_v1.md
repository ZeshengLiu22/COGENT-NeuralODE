# NODE2 Upgrade V1

This architecture reference includes the foundation correctness fixes. Current
launch commands are in [the shell handbook](../handbook.md); optional temporal
objectives are described in [temporal consistency](temporal_consistency.md).

## Overview

Original NODE2 was a latent-space controlled graph Neural ODE:

$$
\begin{aligned}
z_0 &= \operatorname{InitMLP}([\mathrm{hist\_context}, \mathrm{last\_state}, \mathrm{static\_embed}])\\
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, G)\\
\hat y(t) &= \operatorname{DecMLP}(z(t))
\end{aligned}
$$

where $u(t)$ is interpolated future forcing, `s` is the static node embedding, and `G` is the graph.

Upgrade v1 keeps NODE2 as an ODE. Rollout MSE remains the state objective; the current trainer can also add one configurable temporal-consistency term. The four architectural upgrades are:

- residual decoder
- history-aware ODE evolution
- temporal-only Transformer history encoding
- explicit normalized relative time in the NODE2 vector field

The upgraded model config enables all four options. Dataset/protocol scales come from the dataset or protocol config; the model overlay leaves them intact.

## Original NODE2 vs Upgraded NODE2

Original:

$$
\begin{aligned}
z_0 &= \operatorname{InitMLP}([\mathrm{hist\_context}, \mathrm{last\_state}, \mathrm{static\_embed}])\\
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, G)\\
\hat y(t) &= \operatorname{DecMLP}(z(t))
\end{aligned}
$$

Upgraded:

$$
\begin{aligned}
\mathrm{hist\_context} &= \operatorname{LSTM}(\mathrm{step\_embeds})\quad\text{or}\quad \operatorname{Transformer}(\mathrm{step\_embeds})\\
z_0 &= \operatorname{InitMLP}([\mathrm{hist\_context}, \mathrm{last\_state}, \mathrm{static\_embed}])\\
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, \mathrm{hist\_context}, \mathrm{rel\_t}, G)\\
\Delta y(t) &= \operatorname{DecMLP}(z(t))\\
\hat y(t) &= \mathrm{last\_state} + \Delta y(t)
\end{aligned}
$$

Each new term is controlled by a flag. Disabling all upgrade flags restores the old NODE2 behavior as closely as possible:

```yaml
model:
  use_residual_decoder: false
  use_history_in_ode: false
  use_relative_time: false
  history_encoder:
    history_encoder_type: lstm
```

## Upgrade List

### 1. Residual Decoder

Motivation:
The old decoder predicted the full future state directly from latent states. For many rollout problems, the short-term target is closer to "change from the current state" than "absolute state from scratch".

Mathematical idea:

$$
\begin{aligned}
\Delta y(t) &= \operatorname{DecMLP}(z(t))\\
\hat y(t) &= y_{\mathrm{last}} + \Delta y(t)
\end{aligned}
$$

Implementation details:

- Implemented in `models/node2_model.py`.
- Uses `encoded["last_state"]` from `HistoryEncoder`.
- `last_state` has shape `[N_total, F_state]`.
- Decoder output has shape `[N_total, K, F_state]`.
- The residual anchor is added as `last_state.unsqueeze(1)`, broadcasting across future steps.
- The addition happens in normalized model space, matching the existing training target convention.
- The latent decoder MLP can process flattened node-horizon rows in chunks via `model.decoder_chunk_size`. This memory-safety option is independent of whether residual decoding is enabled.

Config:

```yaml
model:
  use_residual_decoder: true
  decoder_chunk_size: 262144
```

How to disable:

```yaml
model:
  use_residual_decoder: false
```

Expected benefit:
This can improve short-horizon stability and make long rollouts easier to learn when future states are smooth increments from the observed history endpoint.

Possible downside:
If the target has large discontinuous jumps, the residual framing may require the decoder to learn larger deltas.

### 2. History-Aware ODE Evolution

Motivation:
Original NODE2 used history only to initialize `z0`. After integration started, the vector field only saw latent state, forcing, static embedding, and graph structure. The upgraded vector field can keep node-wise history context available at every ODE evaluation.

Mathematical idea:

$$
\begin{aligned}
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, c_{\mathrm{hist}}, G)
\end{aligned}
$$

or, when relative time is also enabled:

$$
\begin{aligned}
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, c_{\mathrm{hist}}, \mathrm{rel\_t}, G)
\end{aligned}
$$

Implementation details:

- Implemented in `models/continuous/node_latent_block.py`.
- `LatentNODEFunc` receives `hist_dim` at construction.
- When enabled, `hist_context` is concatenated into the node-wise vector-field input.
- The graph network still performs spatial message passing through the same `edge_index`.
- No cross-attention, hypernetwork, FiLM block, graph-temporal attention, or CDE redesign was added.

Config:

```yaml
model:
  use_history_in_ode: true
```

How to disable:

```yaml
model:
  use_history_in_ode: false
```

Expected benefit:
The ODE can condition its derivative on the summarized past throughout the rollout, not just at initialization.

Possible downside:
The vector-field input dimension grows by `D_hist`, increasing the first dynamics GNN layer size.

### 3. Transformer-Based History Encoder

Motivation:
The old `HistoryEncoder` used a node-wise LSTM over the per-step GNN embeddings. Upgrade v1 adds a temporal-only Transformer so each node can attend across its history window.

Mathematical idea:

$$
\begin{aligned}
h_i^\tau &= \operatorname{GNN}([\mathrm{state}_i^\tau, \mathrm{force}_i^\tau, \mathrm{static\_embed}_i], G)\\
c_i &= \operatorname{Transformer}([h_i^1, \ldots, h_i^H])
\end{aligned}
$$

Implementation details:

- Implemented in `models/encoders/history_encoder.py`.
- The per-time-step spatial GNN remains unchanged.
- The Transformer input is `step_embeds` with shape `[N_total, H, D_enc]`.
- `batch_first=True` is used, so the Transformer attends over dimension `H` for each node independently.
- Sinusoidal positional encoding is added by default.
- `history_context_pooling: last` takes the last temporal token output as `hist_context`.
- `history_context_pooling: mean` is available for mean pooling.
- The LSTM path remains available for ablation, selected only by `history_encoder_type: lstm`. The retired `use_transformer_history` flag is not a runtime selector.
- `hist_context_dim` remains tied to `lstm_hidden_dim`; if Transformer output dimension differs, a small projection maps it to `D_hist`.
- For large node counts, the Transformer runs over node-batch chunks by default. Chunking is along `N_total`, not `H`, so each node still attends across its full history window.
- On CUDA, the temporal Transformer defaults to PyTorch's math scaled-dot-product attention backend to avoid oversized fused-attention launches on large meshes.

Config:

```yaml
model:
  history_encoder:
    history_encoder_type: transformer
    history_transformer_num_layers: 2
    history_transformer_num_heads: 4
    history_transformer_ff_dim: 384
    history_transformer_dropout: 0.05
    history_transformer_chunk_size: 8192
    history_transformer_force_math_sdp: true
    history_use_positional_encoding: true
    history_context_pooling: last
```

How to disable:

```yaml
model:
  history_encoder:
    history_encoder_type: lstm
```

Expected benefit:
Temporal self-attention can model nonlocal dependencies across the known history window more directly than a purely recurrent summary.

Possible downside:
Compute and memory increase with history length because temporal attention has $O(H^2)$ attention cost per node.
Chunking can add some launch overhead, but it lowers peak memory and avoids CUDA kernel configuration failures on large meshes.

### 4. Explicit Relative Time

Motivation:
Original NODE2 received forcing evaluated at continuous solver time, but the vector field did not receive time explicitly. Upgrade v1 adds a scalar relative-time feature while keeping NODE2 as an ODE.

Mathematical idea:

$$
\begin{aligned}
\mathrm{rel\_t} &= t / \mathrm{relative\_time\_scale}\\
\frac{dz}{dt} &= f_\theta(z(t), u(t), s, c_{\mathrm{hist}}, \mathrm{rel\_t}, G)
\end{aligned}
$$

Implementation details:

- Implemented in `models/continuous/node_latent_block.py`.
- `model.relative_time_scale` is a fixed positive finite scale saved with the model config: 180 for standard ISSM, 65 for ANUGA, and 239 for the ISSM paper-matched protocol.
- The scale is independent of the requested horizon, and `rel_t` is not clamped. Extending a rollout therefore preserves its prediction prefix.
- The scalar is broadcast to `[N_total, 1]` and concatenated into the vector-field input.
- The forcing interpolator is unchanged.
- NODE2 remains an ODE, not an NCDE.

Config:

```yaml
model:
  use_relative_time: true
  relative_time_scale: 180  # Standard ISSM; ANUGA: 65, paper-matched ISSM: 239
```

How to disable:

```yaml
model:
  use_relative_time: false
```

Expected benefit:
The vector field can distinguish early-rollout and late-rollout behavior even if latent state and forcing are locally similar.

Possible downside:
The model may learn horizon-position-specific behavior, so extrapolating far outside the training rollout scale should be tested carefully.

## Tensor Shape Walkthrough

The repo uses PyG-batched graph windows with node-major tensors:

```text
x_static:   [N_total, F_static]
state_hist: [N_total, H, F_state]
force_hist: [N_total, H, F_force]
edge_index: [2, E_total]
```

History encoding steps:

1. Static encoding:

```text
static_embed = StaticMLP(x_static)
static_embed: [N_total, D_enc]
```

2. Per-step spatial GNN:

For each history step `tau`:

```text
state_tau = state_hist[:, tau, :]   # [N_total, F_state]
force_tau = force_hist[:, tau, :]   # [N_total, F_force]
step_in   = [state_tau, force_tau, static_embed]
step_h    = GNN(step_in, edge_index)
step_h:   [N_total, D_enc]
```

3. Stack history:

```text
step_embeds = stack(step_h over tau, dim=1)
step_embeds: [N_total, H, D_enc]
```

4. Temporal-only Transformer:

```text
temporal_outputs = Transformer(step_embeds + pos_encoding)
temporal_outputs: [N_total, H, D_enc]
hist_context = temporal_outputs[:, -1, :]
hist_context: [N_total, D_hist]
```

If `D_enc != D_hist`, a projection maps the pooled Transformer output to `D_hist`.

Important distinction:
This is temporal-only attention. It does not jointly attend over node-time pairs. Spatial mixing still happens only through the per-step graph neural network.

## NODE2 Forward Path Walkthrough

1. Run `HistoryEncoder`.

Outputs:

```text
static_embed: [N_total, D_enc]
step_embeds:  [N_total, H, D_enc]
hist_context: [N_total, D_hist]
last_state:   [N_total, F_state]
last_force:   [N_total, F_force]
```

2. Initialize latent state:

```text
z0 = InitMLP([hist_context, last_state, static_embed])
z0: [N_total, D_latent]
```

3. Build forcing interpolant:

```text
control = interpolant([last_force, force_future], [0, t_future])
```

4. Set ODE context:

```text
edge_index
static_embed
hist_context
control
time_scale = t_future[-1]
```

5. Integrate:

```text
rollout = odeint(LatentNODEFunc, z0, [0, t1, ..., tK])
z_future = rollout[1:].permute(1, 0, 2)
z_future: [N_total, K, D_latent]
```

6. Decode:

Original decode path:

```text
y_hat = DecMLP(z_future)
```

Residual decode path:

```text
delta_y = DecMLP(z_future)
y_hat   = last_state.unsqueeze(1) + delta_y
```

## Scope of the Architecture Upgrade

The architecture upgrade did not add the following features:

- graph-temporal attention
- space-time graph Transformer redesign
- horizon-weighted rollout loss
- NODE2-to-NCDE redesign

The current training objective separately supports temporal consistency while
retaining rollout MSE; see [temporal consistency](temporal_consistency.md).

## Ablation Plan

Full upgraded model:

```bash
./train_anuga_node2.sh
```

Residual decoder off:

```yaml
model:
  use_residual_decoder: false
```

History-in-ODE off:

```yaml
model:
  use_history_in_ode: false
```

Transformer history off / LSTM history on:

```yaml
model:
  history_encoder:
    history_encoder_type: lstm
```

Relative time off:

```yaml
model:
  use_relative_time: false
```

All upgrades off:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml ./train_anuga_node2.sh
```

Only residual decoder on:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml ./train_anuga_node2.sh
```

Transformer history on but history-in-ODE off:

```yaml
model:
  use_history_in_ode: false
  history_encoder:
    history_encoder_type: transformer
```

## Backward Compatibility Notes

- `HistoryEncoder.forward` still returns `static_embed`, `step_embeds`, `hist_context`, `last_state`, and `last_force`.
- The previous LSTM history encoder is still available.
- NODE2 can still use the original absolute decoder path.
- NODE2 can still use the original vector field input shape by disabling `use_history_in_ode` and `use_relative_time`.
- Decoder and temporal-Transformer chunking are memory/execution details only. They add no learned parameters and should not require retraining already completed baselines for metric comparison.
- Horizon curriculum is supported separately as upgrade 1.1; see `docs/upgrade_v1.1.md`.
- Subsequent foundation repairs changed dataset inputs, normalization, and relative-time semantics; consult the [foundation report](foundation_correctness_repair.md) for checkpoint compatibility.

Standalone evaluation restores the saved checkpoint config and split. Old ISSM
checkpoints predating the foundation repair require retraining; current CLI
overrides cannot silently change the saved architecture or solver.

## Example Commands

Full upgraded ANUGA NODE2:

```bash
./train_anuga_node2.sh
```

Old baseline-like ANUGA NODE2:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml ./train_anuga_node2.sh
```

Only residual decoder enabled:

```bash
EXTRA_CONFIGS=configs/NODE2_Upgrade1_Ablation/model_node2_residual_only.yaml ./train_anuga_node2.sh
```

Direct Python training with full upgraded configs:

```bash
python scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history8.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node2.yaml
```

Direct Python training with old baseline-like overrides:

```bash
python scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history8.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node2.yaml \
  --config configs/NODE2_Upgrade1_Ablation/model_node2_upgrade_off.yaml
```
