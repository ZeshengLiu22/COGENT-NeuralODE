# NODE2 architecture and the four-factor ablation

This reference describes the active NODE2 implementation in
`models/node2_model.py`, `models/encoders/history_encoder.py`, and
`models/continuous/node_latent_block.py`. Its four independent factors are
Transformer versus LSTM history encoding, residual decoding, history context
in the ODE, and explicit relative time. Full architecture selects Transformer
and enables all three switches.

The scientific input/output contract is unchanged by these choices: H true
history states plus known future forcing produce a complete future state series.
Training predicts $1,\ldots,k_{\mathrm{eff}}$; formal evaluation predicts $S,\ldots,T-1$. State MSE is
always present; optional [TC](temporal_consistency.md) uses the same prediction.

## Static and per-history-step graph encoding

Let $s_i^{raw}$, $y_i^\tau$, and $u_i^\tau$ denote normalized static,
state, and forcing inputs. First embed static features with an MLP:

$$
s_i=E_s(s_i^{raw}).
$$

For every observed history step, apply the same spatial graph network:

$$
h_i^\tau=
G_\phi([y_i^\tau,u_i^\tau,s_i],\mathcal G),
\qquad \tau=1,\ldots,H.
$$

Each GNN call sees one history snapshot on the batch's disconnected graphs.
The output stacks as `[N_total,H,D_enc]`, preserving node identity. The current
model YAML uses GraphSAGE, two layers, GELU, static embedding width 96, and GNN
hidden width 96. Optional dataset edge attributes are not consumed by these
graph layers; connectivity is `edge_index`.

## Factor 1: Transformer or LSTM temporal context

The shorthand temporal-context equation is

$$
c_i=\operatorname{Transformer}(h_i^1,\ldots,h_i^H).
$$

In code, the Transformer first adds sinusoidal positional encodings, applies
two encoder layers, pools the last output token, and projects to the context
width if needed:

$$
c_i=P\left(
\operatorname{Transformer}(h_i^1+p_1,\ldots,h_i^H+p_H)_H
\right).
$$

Attention is across observed time for each node separately. It has no causal
mask because every history state is known; graph message passing supplies
spatial interaction before temporal encoding. Mean pooling is supported by
`history_context_pooling: mean`, while formal full architecture uses `last`.
The canonical Transformer has four heads, feed-forward width 384, and dropout
0.05. Chunking occurs along the node dimension, retaining the full H-token
history per node. The default chunk size is 8192 and CUDA uses math scaled
dot-product attention unless explicitly configured otherwise.

The alternative context is the final top-layer hidden state of a node-wise LSTM:

$$
c_i=\operatorname{LSTM}(h_i^1,\ldots,h_i^H)_{\mathrm{last}}.
$$

`model.history_encoder.history_encoder_type` is the selector:
`transformer` or `lstm`. The code's context width is named `lstm_hidden_dim`
for both encoder choices; it is 96 in the model config. The Transformer uses
an identity projection when its width already equals the context width.

Both choices always initialize the latent state using context, the last
observed normalized state, and static embedding:

$$
z_i(0)=\operatorname{InitMLP}([c_i,y_i^{last},s_i]).
$$

Consequently, turning off history context in the ODE does not disable history
encoding or remove history information from initial conditions.

## Factor 2: history context in the vector field

Without persistent ODE history context, the vector field is

$$
\frac{dz_i}{dt}
=f_\theta(z_i(t),u_i(t),s_i,\mathcal G).
$$

With `model.use_history_in_ode: true`, it is

$$
\frac{dz_i}{dt}
=f_\theta(z_i(t),u_i(t),s_i,c_i,\mathcal G).
$$

These equations omit the independently optional time input for clarity.
Context $c_i$ and static embedding $s_i$ stay fixed during one rollout, while the
latent state and interpolated forcing vary continuously. A graph network
returns one latent derivative per node. Context is concatenated directly into
its input, increasing that input width; there is no separate attention or
conditioning subnetwork in the vector field.

The canonical dynamics network uses latent width 96, GraphSAGE, two layers,
hidden width 96, Softplus, and zero dropout. The topology remains fixed during
integration.

## Factor 3: fixed-scale relative time

With `model.use_relative_time: true`, append the scalar feature

$$
r(t)=t/\tau_{\mathrm{scale}},
$$

broadcast to all nodes. With both optional inputs enabled, the full equation is

$$
\frac{dz_i}{dt}
=f_\theta(z_i(t),u_i(t),s_i,c_i,r(t),\mathcal G).
$$

Here t is solver time relative to the current history anchor. The dataset first
divides adapter time offsets by the trajectory's median positive timestep.
The protocol then supplies a fixed model scale:

| Protocol | `model.relative_time_scale` |
| --- | ---: |
| ISSM | 180 |
| ANUGA | 65 |

The scale must be finite and positive when the feature is enabled. It is saved
with the model and does not depend on K, known_steps, or a requested inference
endpoint. There is no endpoint normalization and no clamp. Changing S reanchors
solver time at a new last-observed state while retaining the same scale.

This removes endpoint dependence from the vector-field time feature.
Prediction-prefix consistency also depends on forcing interpolation and
numerical solver behavior; the real-data prefix audit checks the active path.
It is not a claim that every arbitrary adaptive solver or interpolator is
bitwise prefix-invariant.

Known forcing is interpolated through the last observed forcing at $t=0$ and all
future forcing samples. The canonical interpolation is backward-Hermite cubic;
linear is also supported. Future state never enters interpolation.

## Factor 4: direct or residual decoding

Direct decoding uses

$$
\hat y_i(t)=D_\psi(z_i(t)).
$$

With `model.use_residual_decoder: true`, the decoder predicts a delta:

$$
\Delta\hat y_i(t)=D_\psi(z_i(t)),\qquad
\hat y_i(t)=y_i^{last}+\Delta\hat y_i(t).
$$

All terms are in normalized state space. The anchor has shape
`[N_total,F_state]` and is broadcast across predicted times. Each future delta
is relative to the same observed state, rather than accumulated from earlier
decoded outputs. Residual mode does not impose a hard zero decoder value at
$t=0$; the returned output contains future times only.

The decoder has two hidden layers of width 96 and GELU in the model YAML.
Its optional chunk size (default 262144 flattened node/time rows) controls
memory and is independent of residual mode. Output is
`[N_total,k_eff,F_state]` in training or `[N,T-S,F_state]` for one rollout.

## Integration and numerical precision

`NODE2Model.forward` builds the latent initial condition and integrates
`odeint(f,z0,[0,*t_future])` using standard backpropagation. It drops the initial
latent, decodes every requested future latent, and adds the residual anchor
when enabled. Canonical solver settings are midpoint, backward-Hermite forcing,
rtol 1e-4, atol 1e-5, and no extra ODE options.

Initialization, interpolation, and ODE integration use float32 with autocast
disabled. Encoder and decoder may run under configured training AMP.
Evaluation defaults to float32. Numerical and memory settings are distinct
from the four formal architecture factors.

## Explicit 16-combination experiment

Each dataset has its own `configs/ablations/<dataset>/architecture/` directory.
`full.yaml` explicitly contains:

```yaml
model:
  history_encoder:
    history_encoder_type: transformer
  use_residual_decoder: true
  use_history_in_ode: true
  use_relative_time: true
```

All sixteen numbered files also explicitly set all four choices. The canonical
full architecture matches `a01_transformer_reson_ctxon_timeon.yaml`.

| ID | Encoder | Residual | ODE context | Relative time |
| --- | --- | --- | --- | --- |
| a01 | Transformer | ON | ON | ON |
| a02 | Transformer | ON | ON | OFF |
| a03 | Transformer | ON | OFF | ON |
| a04 | Transformer | ON | OFF | OFF |
| a05 | Transformer | OFF | ON | ON |
| a06 | Transformer | OFF | ON | OFF |
| a07 | Transformer | OFF | OFF | ON |
| a08 | Transformer | OFF | OFF | OFF |
| a09 | LSTM | ON | ON | ON |
| a10 | LSTM | ON | ON | OFF |
| a11 | LSTM | ON | OFF | ON |
| a12 | LSTM | ON | OFF | OFF |
| a13 | LSTM | OFF | ON | ON |
| a14 | LSTM | OFF | ON | OFF |
| a15 | LSTM | OFF | OFF | ON |
| a16 | LSTM | OFF | OFF | OFF |

Phase 2 uses selected $H^*$ from Phase 1, canonical K (ISSM180 / ANUGA64),
canonical S (ISSM60 / ANUGA8), and TC0. The per-scenario epoch budget stays
ISSM60 / ANUGA9. Architecture changes therefore do not change series counts
or the horizon sampling design.

The architecture winner is not propagated to Phase 3 or Phase 4. Both return
to `full.yaml`; Phase 3 scans K and Phase 4 uses canonical K for TC0–TC5.
Only selected $H^*$ propagates. Standalone Phase-2 files under
`launchers/{shell,slurm}/<dataset>/02_architecture/` fail until their
selected-history placeholders are replaced.
