# NODE2 Upgrade V2

> **Historical archive — not the current runtime.** This document describes the
> removed structured V2 experiment. Its model options, paths, and commands are
> retained for historical reference and are not supported by the active NODE2
> pipeline. Archived material is in `legacy-v2/`; use [Upgrade V1](upgrade_v1.md),
> [current launch commands](../handbook.md), and [temporal consistency](temporal_consistency.md) for new runs.

## Overview

The current live NODE2 is already the Upgrade v1 model. It keeps the shared
`HistoryEncoder`, initializes a latent state from history, integrates that
latent state with a controlled graph ODE, and decodes future states. Upgrade v1
also added residual decoding, optional history context inside the ODE,
temporal-only Transformer history encoding, and explicit normalized relative
time inside the NODE2 vector field.

Upgrade v2 keeps that pipeline intact and changes only the NODE2 continuous
latent vector field. Instead of one generic graph network for
`f_theta(z, u, s, c_hist, rel_t, G)`, v2 uses a structured additive vector
field with optional per-branch normalization and fusion:

```text
terms = [f_local, f_spatial, f_forcing, f_coupling]
dz/dt = fuse(norm_each_active_term(terms))
```

Each term is learned and returns `[N_total, D_latent]`. This is still a NODE2
latent ODE, not an NCDE.

## Why A Structured Continuous Block

The handbook treats the continuous block as the natural place for future model
novelty. That is where the model repeatedly reasons about state change under
known forcing and graph structure. Keeping the dataset, history encoder,
interpolation, solver, decoder, and loss unchanged makes the experiment easier
to interpret: any difference between v1 and v2 comes from the vector field.

The v2 decomposition is physics-inspired rather than physics-constrained. It
does not impose a PDE stencil or conservation law. It separates mechanisms that
often matter in simulator rollouts:

- local latent tendency at each node
- graph-mediated spatial propagation
- direct effect of forcing
- interaction among latent state, forcing, and spatial response

## Current V1 NODE2 Vs Upgrade V2 NODE2

Current live v1:

```text
hist_context = LSTM(step_embeds) or Transformer(step_embeds)
z0           = InitMLP([hist_context, last_state, static_embed])
dz/dt        = f_theta(z(t), u(t), s, hist_context, rel_t, G)
delta_y(t)   = DecMLP(z(t))
y_hat(t)     = last_state + delta_y(t)   # if residual decoder enabled
```

Upgrade v2:

```text
ctx          = [static_embed, optional hist_context, optional rel_t]
f_local      = Phi_local([z, ctx])
spatial_feat = GNN_spatial([z, static_embed], G)
f_spatial    = Phi_spatial([spatial_feat, ctx])
f_forcing    = Phi_forcing([u(t), ctx])
f_coupling   = Phi_coupling([z, u(t), spatial_feat, ctx])
dz/dt        = fuse(norm(f_local), norm(f_spatial), norm(f_forcing), norm(f_coupling))
y_hat(t)     = decoder(z(t))   # residual mode unchanged from v1 if enabled
```

## Term By Term

### Local Term

Intuition: captures node-local latent self-dynamics conditioned on static
features, history summary, and relative rollout time when those v1 options are
enabled.

Mathematical role:

```text
f_local = Phi_local([z, ctx])
```

Implementation shape:

```text
input:  [N_total, D_latent + D_ctx]
output: [N_total, D_latent]
```

If disabled with `structured_dynamics.use_f_local: false`, the contribution is
a zero tensor with shape `[N_total, D_latent]`.

### Spatial Term

Intuition: captures graph-mediated propagation and neighbor interaction.

Mathematical role:

```text
spatial_feat = GNN_spatial([z, static_embed], edge_index)
f_spatial    = Phi_spatial([spatial_feat, ctx])
```

Implementation shape:

```text
GNN input:  [N_total, D_latent + D_static_embed]
GNN output: [N_total, D_latent]
MLP input:  [N_total, D_latent + D_ctx]
MLP output: [N_total, D_latent]
```

If disabled with `structured_dynamics.use_f_spatial: false`, the contribution
is zero. The spatial GNN may still run if the coupling term is enabled, because
coupling uses `spatial_feat`.

### Forcing Term

Intuition: captures direct external-forcing effects on latent evolution.

Mathematical role:

```text
f_forcing = Phi_forcing([u(t), ctx])
```

Implementation shape:

```text
input:  [N_total, F_force + D_ctx]
output: [N_total, D_latent]
```

If disabled with `structured_dynamics.use_f_forcing: false`, the contribution
is zero.

### Coupling Term

Intuition: captures interactions between current latent state, continuous
forcing, and graph-mediated spatial response.

Mathematical role:

```text
f_coupling = Phi_coupling([z, u(t), spatial_feat, ctx])
```

Implementation shape:

```text
input:  [N_total, D_latent + F_force + D_latent + D_ctx]
output: [N_total, D_latent]
```

If disabled with `structured_dynamics.use_f_coupling: false`, the contribution
is zero.

## Branch Normalization And Fusion

Structured v2 can control derivative scale before returning `dz/dt`. This is
important because the raw sum of four learned branches can have a larger
initial or learned magnitude than the v1 vector field.

`structured_dynamics.term_norm` applies independently to each active branch
over the latent channel dimension:

- `none`: no branch normalization
- `layernorm`: `nn.LayerNorm(D_latent)`
- `rmsnorm`: `nn.RMSNorm(D_latent)`

`structured_dynamics.fusion` combines active branches after optional
normalization:

- `sum`: raw sum, matching the original structured-v2 behavior
- `mean`: average over active branches with no learned gates
- `direct_gated`: learned scalar gate per branch
- `softmax_gated`: learned scalar logits over active branches, softmaxed so
  active weights are positive and sum to one

`structured_dynamics.gate_init: active_mean` initializes direct gates to
`1 / num_active_terms`. With four active terms this starts at `0.25` per term;
with coupling disabled it starts the three active terms at `1/3`. For
`softmax_gated`, zero logits produce the same equal active weights at
initialization, but the logits remain independent learnable parameters.

## Interaction With V1 Upgrades

Transformer history remains available through the existing `HistoryEncoder`.
The v2 block consumes only the stable encoder outputs already used by v1:
`static_embed`, `hist_context`, `last_state`, and `last_force`.

`use_history_in_ode` still controls whether `hist_context` enters the vector
field. In v2, it enters through `ctx`.

`use_relative_time` still controls whether normalized relative time enters the
vector field. In v2, it enters through `ctx`.

Residual decoding is unchanged. `NODE2Model` still adds
`encoded["last_state"].unsqueeze(1)` to decoder outputs when
`use_residual_decoder` is true.

## What Is Not Implemented

Upgrade v2 does not add:

- NCDE redesign
- graph-temporal attention
- slow/fast dual-latent branches
- new loss functions
- horizon-weighted objectives
- rollout consistency objectives
- explicit scenario conditioning
- wet/dry mask objectives

## Config Flags

Old configs that omit `node2_vector_field_type` still use the v1 block:

```yaml
model:
  node2_vector_field_type: v1_base
```

The v2 overlay enables the structured block:

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

The committed overlay is:

```text
configs/model_node2_upgrade2.yaml
```

No ablation config files are required for normal v2 use.

## Ablation Plan

Useful comparison points:

- v1 base block: omit `configs/model_node2_upgrade2.yaml`
- full structured v2: enable all four terms
- structured v2 without coupling: set `use_f_coupling: false`
- structured v2 without forcing: set `use_f_forcing: false`
- structured v2 without spatial: set `use_f_spatial: false`
- local-only sanity check: keep only `use_f_local: true`

For one-off ablations, add a temporary overlay after
`configs/model_node2_upgrade2.yaml`. For example:

```yaml
model:
  structured_dynamics:
    use_f_coupling: false
```

The repo does not need permanent ablation-study configs for these switches.

## Backward Compatibility Notes

The old v1 path remains implemented by `LatentNODEFunc`.
`NODE2Model` chooses the vector field from `model.node2_vector_field_type`.
Missing selector values default to `v1_base`, so existing configs and old
training commands keep their live v1 behavior.

Existing repo-specific defaults were preserved. Upgrade v2 adds only the new
selector and structured-dynamics fields, and v2 is activated by the new overlay
config rather than by editing `configs/model_node2.yaml`.

## Example Commands

V1 ANUGA run:

```bash
bash train_anuga_node2.sh
```

Full v2 ANUGA run:

```bash
bash train_anuga_node2_upgrade2.sh
```

Equivalent explicit command:

```bash
python scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node2.yaml \
  --config configs/model_node2_upgrade2.yaml \
  --run-name anuga_node2_upgrade2_manual
```

Horizon curriculum is a shared trainer feature from upgrade 1.1 and can be
combined with upgrade-v2 configs; see `docs/upgrade_v1.1.md`.

V2 without coupling, using a temporary overlay:

```bash
python scripts/train.py \
  --config configs/ANUGA_History_Scan/base_ANUGA_history1.yaml \
  --config configs/anuga.yaml \
  --config configs/model_node2.yaml \
  --config configs/model_node2_upgrade2.yaml \
  --config /path/to/model_node2_v2_no_coupling.yaml \
  --run-name anuga_node2_v2_no_coupling
```

V2 without forcing uses the same pattern with:

```yaml
model:
  structured_dynamics:
    use_f_forcing: false
```

V2 with LSTM history instead of Transformer can use a final temporary overlay:

```yaml
model:
  history_encoder:
    history_encoder_type: lstm
    use_transformer_history: false
    history_use_positional_encoding: false
```

For Slurm, use:

```bash
sbatch sbatch_scripts/train_node2_upgrade2.sh
```

For ISSM, use:

```bash
bash train_issm_node2_upgrade2.sh
```
