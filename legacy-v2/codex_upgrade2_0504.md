You are modifying an existing continuous-time graph emulator project whose current live version is **already after NODE2 Upgrade v1**.

That means the repo already contains the v1 NODE2 architectural upgrades described in `upgrade_v1.md`, namely:

- residual decoder
- history-aware ODE evolution
- temporal-only Transformer history encoder
- explicit relative time in the NODE2 vector field

Your task is to implement **Upgrade v2 only**, on top of that current live codebase, with minimal disruption and strong backward safety.

IMPORTANT: Before making any code changes, first read the current repo carefully. This is an existing project folder with user-modified configs and experiment-specific values. Respect the current repo state. Do not reset anything back to handbook defaults.

==================================================
PRIMARY DESIGN REFERENCES
==================================================

Use the following files as the design reference:

1. the original continuous graph emulator handbook
2. `upgrade_v1.md`
3. the current repo state itself

Use the handbook as an architectural reference and `upgrade_v1.md` as the reference for the current live NODE2 behavior. But do NOT treat either document as permission to reset current repo values.

The handbook establishes these key contracts and principles:

1. Shared `HistoryEncoder` output contract should remain stable whenever possible:
   - `static_embed`: [N_total, D_enc]
   - `step_embeds`:  [N_total, H, D_enc]
   - `hist_context`: [N_total, D_hist]
   - `last_state`:   [N_total, F_state]
   - `last_force`:   [N_total, F_force]

2. NODE2 remains a **latent ODE**, not an NCDE.

3. The continuous block is the most natural place for future model novelty.

4. Version-1 / version-2 training should remain simple unless explicitly requested otherwise:
   - rollout MSE only
   - no new training objectives
   - no horizon-weighted loss
   - no rollout consistency loss

5. Keep the unified dataset/model/training framework intact wherever possible.

`upgrade_v1.md` establishes the current live NODE2 conceptual form:

```text
hist_context = LSTM(step_embeds) or Transformer(step_embeds)
z0           = InitMLP([hist_context, last_state, static_embed])
dz/dt        = f_theta(z(t), u(t), s, hist_context, rel_t, G)
delta_y(t)   = DecMLP(z(t))
y_hat(t)     = last_state + delta_y(t)
```

==================================================
CRITICAL PROJECT-SAFETY RULE
==================================================

This task is being applied to an existing project folder that may already contain user-modified configs, args, script defaults, and experiment-specific values from previous runs.

Therefore:

1. If an existing config/arg value in the current codebase differs from a handbook or old prompt default, KEEP the existing project value.
2. Do NOT change an existing arg/value back to a handbook default just because the handbook mentions that default.
3. Only add new args/fields when needed for Upgrade v2.
4. For newly added args, use the default behavior requested in this prompt.
5. Preserve all unrelated existing values exactly as they are unless a change is strictly required for compatibility.
6. Treat the current live repo state as the source of truth for already-existing args.

==================================================
HIGH-LEVEL GOAL OF UPGRADE V2
==================================================

Implement a **structured physics-inspired latent vector field** for NODE2, while preserving the rest of the current v1 pipeline as much as possible.

Upgrade v2 should keep:

- the shared `HistoryEncoder`
- the current NODE2 initialization path
- the current forcing interpolation logic
- the current ODE solver path
- the current decoder behavior, including optional residual decoding from v1
- the current training/evaluation objectives and workflow

Upgrade v2 should change only the NODE2 continuous block design, replacing the current generic latent vector field with a structured one.

Target conceptual form:

```text
dz/dt = f_local + f_spatial + f_forcing + f_coupling
```

where each term is a learnable neural component and all terms produce tensors of shape:

```text
[N_total, D_latent]
```

This is a NODE2-only continuous-block upgrade. Do NOT redesign NODE1 or NCDE1.

==================================================
IMPORTANT DESIGN DECISION
==================================================

For project safety, DO NOT fold the new structured dynamics into the existing v1 `LatentNODEFunc` behind a large pile of conditional logic unless the repo structure absolutely forces that.

Preferred design:

1. Keep the current v1 `LatentNODEFunc` behavior intact as the existing baseline path.
2. Add a **separate** structured NODE2 continuous block class for Upgrade v2, for example:
   - `StructuredLatentNODEFunc`
3. In `NODE2Model`, select between:
   - existing v1-style `LatentNODEFunc`
   - new `StructuredLatentNODEFunc`
   using a new config selector.

Reason:
- this is safer for the live codebase
- this preserves the v1 baseline path cleanly
- this makes ablations and reproducibility much easier
- this avoids making the old block unreadable

==================================================
RECOMMENDED NEW CONFIG SELECTOR
==================================================

Add a NODE2-specific vector-field selector such as:

```yaml
model:
  node2_vector_field_type: structured_v2
```

Allowed choices should include at least:

```yaml
model:
  node2_vector_field_type: v1_base
```

and

```yaml
model:
  node2_vector_field_type: structured_v2
```

Rules:

1. Existing configs that do not specify this new field should continue to behave like the current live v1 code as closely as possible.
2. Do NOT silently switch existing old configs to the new vector field.
3. Put the Upgrade v2 default in a **new overlay config** rather than rewriting the old baseline config behavior.
4. The recommended run style for v2 should be:
   - existing dataset/base config(s)
   - existing `model_node2.yaml`
   - plus a new `model_node2_upgrade2.yaml` overlay

This is strongly preferred over changing the old `model_node2.yaml` in a way that surprises existing experiments.

==================================================
WHAT UPGRADE V2 SHOULD IMPLEMENT
==================================================

Implement a structured NODE2 continuous block whose derivative is decomposed as:

```text
dz/dt = f_local(z, context) + f_spatial(z, context, G) + f_forcing(u, context) + f_coupling(z, u, context, G)
```

where `context` may include:

- `static_embed`
- optional `hist_context` from v1
- optional `rel_t` from v1

The goal is not to impose hard physics constraints. This is a **physics-inspired / mechanism-aware decomposition**, not a strict PDE solver.

==================================================
FORMAL TARGET DESIGN
==================================================

The new structured vector field should be conceptually consistent with the current live v1 NODE2 pipeline.

Let:

- `z(t)` be the latent state
- `u(t)` be interpolated future forcing at continuous time `t`
- `s` be static embedding
- `c_hist` be optional node-wise history context from v1
- `rel_t` be optional relative time scalar from v1
- `G` be the graph structure

Define a shared node-wise context tensor:

```text
ctx = concat([s, optional c_hist, optional rel_t])
```

Then define:

```text
f_local    = Phi_local([z, ctx])
f_spatial  = Phi_spatial([GNN_spatial([z, s], G), ctx])
f_forcing  = Phi_forcing([u(t), ctx])
f_coupling = Phi_coupling([z, u(t), GNN_spatial([z, s], G), ctx])
```

and combine them by simple summation:

```text
dz/dt = f_local + f_spatial + f_forcing + f_coupling
```

All four terms should map to:

```text
[N_total, D_latent]
```

The combination should stay simple:
- sum the component outputs directly
- no attention fusion
- no hypernetwork
- no FiLM conditioning for this upgrade
- no slow/fast dual-latent redesign

==================================================
MINIMALISM RULE
==================================================

Keep Upgrade v2 focused and local.

Do NOT implement any of the following in v2:

- new loss functions
- new rollout/training objectives
- NCDE redesign
- slow/fast dual-latent ODE
- graph-temporal attention
- space-time Transformer redesign
- explicit scenario conditioning
- wet/dry mask objectives
- large refactors of unrelated modules

This is a structured continuous-block upgrade only.

==================================================
IMPLEMENTATION PREFERENCE FOR THE NEW BLOCK
==================================================

Preferred implementation strategy:

1. Reuse as much of the existing v1 `LatentNODEFunc` design pattern as is reasonable.
2. Create a new class such as `StructuredLatentNODEFunc` in the continuous module area.
3. Give it the same external calling style expected by `odeint`.
4. Keep its context-registration / setter pattern compatible with the current model code if such a pattern already exists.
5. Use clear submodules for the four components:
   - `local_mlp`
   - `spatial_gnn` + `spatial_out_mlp`
   - `forcing_mlp`
   - `coupling_mlp`

If the current `LatentNODEFunc` already uses a graph block plus MLP head, then follow that style closely.

==================================================
TERM-BY-TERM REQUIREMENTS
==================================================

### 1. `f_local`

Purpose:
- capture node-local latent self-dynamics and static/history/time-conditioned tendencies

Recommended input:

```text
[z, ctx]
```

Recommended implementation:
- small MLP
- output dimension = `D_latent`

### 2. `f_spatial`

Purpose:
- capture graph-mediated spatial propagation / neighborhood interaction

Recommended path:

```text
spatial_feat = GNN_spatial([z, s], edge_index)
f_spatial    = Phi_spatial([spatial_feat, ctx])
```

Notes:
- use the same graph / `edge_index`
- keep it simple and stable
- if the current continuous block already has a good graph operator, reuse its style

### 3. `f_forcing`

Purpose:
- capture direct external-forcing contribution to latent evolution

Recommended input:

```text
[u_t, ctx]
```

Recommended implementation:
- small MLP
- output dimension = `D_latent`

### 4. `f_coupling`

Purpose:
- capture interaction between current latent state, forcing, and spatial response

Recommended input:

```text
[z, u_t, spatial_feat, ctx]
```

Recommended implementation:
- small MLP
- output dimension = `D_latent`

==================================================
COMPONENT ABLATION FLAGS
==================================================

Add optional ablation flags for the structured block:

```yaml
model:
  structured_dynamics:
    use_f_local: true
    use_f_spatial: true
    use_f_forcing: true
    use_f_coupling: true
```

Rules:

1. These flags matter only when `node2_vector_field_type: structured_v2`.
2. When a component is disabled, replace it with a zero contribution of the correct shape.
3. Keep the combination rule simple and deterministic.
4. This supports clean ablations without modifying code again.

==================================================
INTERACTION WITH V1 FEATURES
==================================================

Upgrade v2 is built **on top of** the current v1 NODE2 features, not instead of them.

Therefore, the new structured block must be compatible with these existing v1 options:

1. `use_history_in_ode`
2. `use_relative_time`
3. residual decoder
4. Transformer or LSTM history encoder

Interpretation:

- if `use_history_in_ode = true`, `hist_context` should be included in `ctx`
- if `use_history_in_ode = false`, `hist_context` should not be injected into the vector field terms
- if `use_relative_time = true`, `rel_t` should be included in `ctx`
- if `use_relative_time = false`, do not include it
- residual decoding remains handled in `NODE2Model`, unchanged conceptually from v1

Do NOT remove or rewrite the v1 features unless a tiny compatibility adjustment is required.

==================================================
SHAPE CONTRACTS TO PRESERVE
==================================================

Preserve the node-major PyG-batched shape conventions.

Important shapes:

```text
x_static:      [N_total, F_static]
state_hist:    [N_total, H, F_state]
force_hist:    [N_total, H, F_force]
force_future:  [N_total, K, F_force]
last_state:    [N_total, F_state]
last_force:    [N_total, F_force]
static_embed:  [N_total, D_enc]
hist_context:  [N_total, D_hist]
z(t):          [N_total, D_latent]
u(t):          [N_total, F_force]
rel_t:         [N_total, 1]
ctx:           [N_total, D_ctx]
dz/dt:         [N_total, D_latent]
```

Document all concatenation shapes with inline comments.

==================================================
CONFIG / CLI REQUIREMENTS
==================================================

Add only the new args needed for Upgrade v2.

Recommended new fields:

```yaml
model:
  node2_vector_field_type: structured_v2
  structured_dynamics:
    use_f_local: true
    use_f_spatial: true
    use_f_forcing: true
    use_f_coupling: true
```

Optional additional fields only if truly needed for clean implementation:

```yaml
model:
  structured_dynamics:
    spatial_out_hidden_dim: <int>
    forcing_hidden_dim: <int>
    coupling_hidden_dim: <int>
    local_hidden_dim: <int>
```

But keep this minimal. Prefer reusing existing model hidden sizes where practical.

Rules:

1. Do NOT add a large number of new knobs unless necessary.
2. Wire new args through the existing config system.
3. Preserve existing CLI/YAML override behavior.
4. Prefer a new overlay config file for Upgrade v2.

==================================================
PREFERRED FILE STRATEGY
==================================================

Update all relevant files, but keep the v2 diff local.

Expected files to touch include, if they exist in the repo:

- `models/continuous/node_latent_block.py`
- or a new file such as `models/continuous/node_latent_block_structured.py`
- `models/node2_model.py`
- model factory / registry if present
- relevant config files
- relevant `.sh` scripts
- documentation

Preferred documentation strategy:

1. Keep `upgrade_v1.md` as the reference for v1.
2. Create a **new** markdown doc:
   - `upgrade_v2.md`
3. Create/update the Codex instruction file for this task:
   - `codex_upgrade2_0504.md`

Do NOT overwrite the meaning of `upgrade_v1.md`.

==================================================
CONFIG FILE POLICY
==================================================

Strong preference:

1. Leave existing `model_node2.yaml` behavior as close as possible to the current live v1 behavior.
2. Add a new overlay config such as:
   - `configs/model_node2_upgrade2.yaml`
3. Put the Upgrade v2 selector and new structured-dynamics args there.
4. If needed, add additional ablation overlays, but keep them minimal.

This avoids breaking existing runs.

==================================================
UPDATE SHELL SCRIPTS CAREFULLY
==================================================

Update only the relevant NODE2 shell scripts, and do so conservatively.

Rules:

1. Do NOT overwrite unrelated experiment-specific values.
2. Keep old v1-style commands easy to run.
3. Add a clean way to run v2, preferably by appending an extra config overlay.

Preferred pattern:

```bash
python scripts/train.py \
  --config configs/base_*.yaml \
  --config configs/<dataset>.yaml \
  --config configs/model_node2.yaml \
  --config configs/model_node2_upgrade2.yaml
```

If the repo already uses launcher scripts with `EXTRA_CONFIGS`, follow that style.

==================================================
DOCUMENTATION REQUIREMENT: CREATE upgrade_v2.md
==================================================

Create a substantial markdown file named:

- `upgrade_v2.md`

This should explain Upgrade v2 clearly for a new lab member.

Required contents:

1. Overview
   - current live v1 NODE2 behavior
   - what Upgrade v2 adds

2. Why a structured continuous block
   - motivation for moving novelty into the NODE2 vector field
   - why this is more natural than changing the dataset/history/training pipeline

3. Current v1 NODE2 vs Upgrade v2 NODE2

Explicitly compare:

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
ctx         = [static_embed, optional hist_context, optional rel_t]
f_local     = Phi_local([z, ctx])
spatial_feat= GNN_spatial([z, static_embed], G)
f_spatial   = Phi_spatial([spatial_feat, ctx])
f_forcing   = Phi_forcing([u(t), ctx])
f_coupling  = Phi_coupling([z, u(t), spatial_feat, ctx])
dz/dt       = f_local + f_spatial + f_forcing + f_coupling
y_hat(t)    = decoder(z(t))   # residual mode unchanged from v1 if enabled
```

4. Term-by-term explanation
   - local term
   - spatial term
   - forcing term
   - coupling term

For each term include:
- intuition
- mathematical role
- implementation input/output shape
- what happens if the term is disabled

5. Interaction with v1 upgrades
   - Transformer history remains available
   - history-in-ODE still matters through `ctx`
   - relative time still matters through `ctx`
   - residual decoder stays unchanged

6. What is NOT implemented
   - no NCDE redesign
   - no graph-temporal attention
   - no slow/fast branch design
   - no new losses
   - no scenario conditioning

7. Config flags
   - `node2_vector_field_type`
   - `structured_dynamics.use_f_local`
   - `structured_dynamics.use_f_spatial`
   - `structured_dynamics.use_f_forcing`
   - `structured_dynamics.use_f_coupling`

8. Ablation plan
   Include at least:
   - v1 base block
   - full structured_v2
   - structured_v2 without coupling
   - structured_v2 without forcing
   - structured_v2 without spatial
   - local-only sanity check if practical

9. Backward compatibility notes
   - old v1 path remains intact
   - existing configs preserved
   - new v2 should be activated via overlay config

10. Example commands
   Include examples for:
   - v1 run
   - full v2 run
   - v2 without coupling
   - v2 without forcing
   - v2 but using LSTM history instead of Transformer if possible

11. Existing-config preservation rule
   Explicitly state that old repo-specific defaults were preserved and only newly introduced Upgrade v2 args were added.

==================================================
SCOPE CONTROL RULE
==================================================

This is an Upgrade v2 task focused only on the NODE2 continuous block and its direct plumbing.

Allowed changes:
- NODE2 continuous block implementation
- minimal NODE2 model selection logic
- config plumbing for selecting v1 vs v2
- NODE2-related shell script updates
- documentation for upgrade v2

Not allowed unless strictly necessary:
- redesign of NODE1
- redesign of NCDE1
- changes to training objectives
- unrelated config cleanup
- dataset redesign
- broad refactors

If shared files must change, preserve backward compatibility.

==================================================
REQUIRED WORKFLOW
==================================================

Before coding, inspect and understand at minimum:

1. current `NODE2Model`
2. current `LatentNODEFunc`
3. the current v1 config/plumbing
4. current `HistoryEncoder` outputs actually used by NODE2
5. config loading / model factory path
6. NODE2-related `.sh` launcher scripts
7. any current docs related to NODE2 upgrades

Then implement Upgrade v2 carefully.

==================================================
SELF-CHECK BEFORE FINISHING
==================================================

Before completing the task, verify:

1. old v1 path still exists and still constructs correctly
2. new structured_v2 path constructs correctly
3. `NODE2Model` chooses the correct block from config
4. disabling all structured terms behaves sensibly or is rejected clearly if that case is unsupported
5. structured component outputs all have shape `[N_total, D_latent]`
6. history/time injection still works correctly under v1 flags
7. residual decoding behavior is unchanged from v1
8. existing unrelated config values were preserved
9. new v2 run is activated by overlay config rather than silently changing old runs
10. documentation is complete and accurate

If tests or smoke tests exist, run them.

==================================================
OUTPUT FORMAT AT THE END
==================================================

At the end of your work, provide:

1. a concise summary of what was changed
2. a list of modified files
3. any assumptions made
4. any follow-up issues or TODOs
5. confirmation that existing repo-specific defaults were preserved wherever possible
