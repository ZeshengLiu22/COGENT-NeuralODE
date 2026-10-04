You are modifying an existing continuous-time graph emulator project that already contains:

- a shared `HistoryEncoder`
- `NODE1Model`, `NODE2Model`, `NCDE1Model`
- `StateSpaceNODEFunc`, `LatentNODEFunc`, `LatentNCDEFunc`
- configs, scripts, and training/evaluation code

Your task is to implement a NODE2 upgrade package focused on architecture only.

IMPORTANT: Before making any code changes, first read all relevant files in the repo and understand the current implementation. This is an existing project folder that may already contain user-modified configs and experiment-specific values. Respect the current repo state.

==================================================
PRIMARY DESIGN REFERENCE
==================================================

Use the project’s original continuous graph emulator handbook as the architectural/design reference, but NOT as an instruction to reset existing repo values back to handbook defaults.

The handbook establishes the following important contracts and design assumptions:

1. Shared `HistoryEncoder` input contract:
   - `x_static`:   [N_total, F_static]
   - `state_hist`: [N_total, H, F_state]
   - `force_hist`: [N_total, H, F_force]
   - `edge_index`: [2, E_total]

2. Shared `HistoryEncoder` output contract:
   - `static_embed`: [N_total, D_enc]
   - `step_embeds`:  [N_total, H, D_enc]
   - `hist_context`: [N_total, D_hist]   # previously D_lstm in handbook; now keep backward-compatible naming where practical
   - `last_state`:   [N_total, F_state]
   - `last_force`:   [N_total, F_force]

3. Original NODE2 definition:
   - `z0 = InitMLP([hist_context, last_state, static_embed])`
   - latent dynamics: `dz/dt = f_theta(z(t), u(t), s, G)`
   - decoder: `y_hat(t) = DecMLP(z(t))`

4. NODE2 is a latent ODE, not an NCDE. Keep NODE2 as an ODE with forcing interpolation. Do NOT redesign NODE2 into a CDE.

5. Version-1 training philosophy should remain simple:
   - rollout MSE only
   - no new training objectives in this upgrade
   - no horizon-weighted loss yet
   - no rollout-consistency loss yet

==================================================
CRITICAL PROJECT-SAFETY RULE
==================================================

This task is being applied to an existing project folder that may already contain user-modified configs, args, script defaults, and experiment-specific values from previous runs.

Therefore:

1. If an existing config/arg value in the current codebase differs from the original design-handbook default, KEEP the existing project value.
2. Do NOT change an existing arg/value back to the handbook default just because the handbook mentions that default.
3. Only add new args/fields when needed for the requested upgrades.
4. For newly added args, use the requested default behavior from this prompt.
5. When touching config files, preserve all unrelated existing values exactly as they are unless a change is strictly required for compatibility.
6. If there is any mismatch between handbook defaults and the current repo’s already-edited defaults, treat the current repo state as the source of truth for existing args, and use the handbook only as an architectural/design reference.

==================================================
GOAL OF THIS UPGRADE
==================================================

Implement the following four NODE2-related upgrades, each controlled by config/CLI flags, with the default behavior set to ENABLED for newly introduced flags:

(1) Residual decoder
(2) Make NODE2 history-aware during evolution
(3) Strengthen the HistoryEncoder by replacing node-wise LSTM with a Transformer over history
(4) Add relative time explicitly to the NODE2 vector field

Do NOT implement the following yet:
- horizon-weighted rollout loss
- rollout consistency regularization
- graph-temporal attention
- space-time graph transformer redesign
- NCDE redesign of NODE2

Focus only on architecture changes and required plumbing/documentation.

==================================================
SCOPE CONTROL RULE
==================================================

This is a NODE2-focused upgrade.

Allowed changes:
- `HistoryEncoder` internals
- `NODE2Model`
- `LatentNODEFunc`
- decoder behavior
- config / arg plumbing
- shell scripts
- documentation
- lightweight utility helpers if required

Not allowed unless strictly necessary for compatibility:
- conceptual redesign of `NODE1Model`
- conceptual redesign of `NCDE1Model`
- changes to training objectives
- unrelated refactors
- changes that alter existing experiment defaults unrelated to this task

If shared modules are changed, preserve backward compatibility as much as possible for NODE1 and NCDE1.

==================================================
REQUIRED WORKFLOW
==================================================

Before coding, inspect and understand at minimum:

1. the current `HistoryEncoder`
2. `NODE2Model`
3. `LatentNODEFunc`
4. decoder implementation(s)
5. argument/config loading path
6. training and evaluation scripts
7. all relevant YAML/JSON config files
8. `.sh` launcher scripts
9. any current docs/README related to NODE2

Then implement the upgrades below carefully.

==================================================
UPGRADE 1: RESIDUAL DECODER
==================================================

Add an optional residual decoder mode for NODE2.

Desired behavior:

- Original mode:
    y_hat(t) = DecMLP(z(t))

- Residual mode:
    delta_y(t) = DecMLP(z(t))
    y_hat(t)   = y_last + delta_y(t)

Use the last observed state from the history window as the residual anchor:
- `last_state` from `HistoryEncoder`

Requirements:

1. Residual decoder must be optional.
2. Default for the new flag should be enabled.
3. Preserve the original non-residual decode path when disabled.
4. Ensure tensor shapes match exactly:
   - `last_state`: [N_total, F_state]
   - decoded rollout typically: [N_total, K_eff_or_Kmax, F_state] or equivalent stacked future form
   - broadcast/add residual anchor correctly across future steps
5. Be careful if the existing decoder operates on flattened latent states or reshaped future trajectories.
6. Respect any existing normalized-space conventions. Do not silently move this into physical space if the model currently predicts in normalized space.

New flag:
- `use_residual_decoder: bool = true`

==================================================
UPGRADE 2: MAKE NODE2 HISTORY-AWARE DURING EVOLUTION
==================================================

Current NODE2 uses history mainly to form `z0`.

Upgrade NODE2 so the latent vector field can also consume history context at every ODE function evaluation.

Target conceptual form:

    dz/dt = f_theta(z(t), u(t), s, c_hist, rel_t, G)

where:
- `z(t)` is the evolving latent state
- `u(t)` is interpolated future forcing at continuous time t
- `s` is static embedding
- `c_hist` is node-wise history context from `HistoryEncoder`
- `rel_t` is optional relative time input
- `G` is graph structure

Requirements:

1. Make this history-in-ODE behavior optional.
2. Default for the new flag should be enabled.
3. Preserve the original vector-field path when disabled.
4. Prefer the simplest stable implementation:
   - inject `c_hist` by concatenation into the node-wise vector-field input
5. Do not overcomplicate with cross-attention, FiLM, hypernetworks, or graph-temporal attention in this version unless the repo structure absolutely demands a tiny conditioning block.
6. If the current `LatentNODEFunc` already separates different input streams internally, extend that structure minimally and clearly.

New flag:
- `use_history_in_ode: bool = true`

==================================================
UPGRADE 3: STRONGER HISTORYENCODER WITH TEMPORAL-ONLY TRANSFORMER
==================================================

Replace the current node-wise LSTM history summarization with an optional Transformer-based temporal history encoder.

This must be TEMPORAL-ONLY attention, not graph-temporal attention.

Very important:
- spatial interaction should still come from the per-time-step GNN
- the Transformer should model dependencies only along the history/time dimension for each node independently

--------------------------------------------------
TRUE SHAPE CONTRACT TO FOLLOW
--------------------------------------------------

The repo uses PyG-batched graph windows. Therefore the true shape convention is node-major, not batch-major.

Inputs to `HistoryEncoder`:
- `x_static`:   [N_total, F_static]
- `state_hist`: [N_total, H, F_state]
- `force_hist`: [N_total, H, F_force]

Required intermediate/output contract:
- `static_embed`: [N_total, D_enc]
- `step_embeds`:  [N_total, H, D_enc]
- `hist_context`: [N_total, D_hist]

This means:

1. For each history step tau in {0, ..., H-1}:
   - take `state_hist[:, tau, :]` -> [N_total, F_state]
   - take `force_hist[:, tau, :]` -> [N_total, F_force]
   - concatenate with `static_embed` -> [N_total, F_state + F_force + D_enc]
   - run the per-step spatial GNN using the same `edge_index`
   - obtain one step embedding -> [N_total, D_enc]

2. Stack all per-step embeddings across time:
   - `step_embeds`: [N_total, H, D_enc]

3. Run temporal self-attention across the H dimension for each node independently.
   The intended conceptual sequence for one node i is:
   - `[h_i^1, h_i^2, ..., h_i^H]`

4. Produce:
   - `hist_context`: [N_total, D_hist]

--------------------------------------------------
IMPLEMENTATION PREFERENCE FOR THE TRANSFORMER
--------------------------------------------------

Use one of these clean implementations:

Preferred option:
- use `nn.TransformerEncoderLayer(..., batch_first=True)`
- input directly as `[N_total, H, D_enc]`
- apply temporal self-attention over dimension H
- output `[N_total, H, D_hist_or_D_enc]`
- take either:
  a) last token output `[:, -1, :]`, or
  b) temporal mean pooling
- choose one and document it clearly
- prefer last-token output for forecasting consistency unless the existing repo structure strongly favors mean pooling

Alternative acceptable option:
- transpose to `[H, N_total, D_enc]` if `batch_first=False` is already the project style

Do NOT introduce an unnecessary fake batch dimension such as `(B, N, H, D)` unless the repo already reconstructs true per-graph mini-batch dimensions explicitly.
The canonical contract here is node-major PyG batched:
- `[N_total, H, D]`

--------------------------------------------------
POSITIONAL INFORMATION
--------------------------------------------------

Add positional encoding over the history steps H.

Requirements:
1. positional encoding must be optional
2. default enabled
3. sinusoidal or learned positional encoding is acceptable
4. keep implementation simple and well documented

--------------------------------------------------
BACKWARD COMPATIBILITY
--------------------------------------------------

Keep the previous LSTM history path available for ablation.

New flags / config:
- `history_encoder_type: str = "transformer"`   # choices should include at least ["transformer", "lstm"]
- `use_transformer_history: bool = true`
- `history_transformer_num_layers`
- `history_transformer_num_heads`
- `history_transformer_ff_dim`
- `history_transformer_dropout`
- `history_use_positional_encoding: bool = true`
- optional: `history_context_pooling: str = "last"`  # choices ["last", "mean"] if helpful

Rules:
1. If `history_encoder_type == "lstm"` or transformer is disabled, preserve previous LSTM behavior.
2. If transformer is enabled, make it the default path.
3. Preserve the external `HistoryEncoder` output contract as much as possible.
4. Update comments/docstrings to clarify that `hist_context` may now come from LSTM or Transformer.
5. Avoid unnecessary memory blowup.

==================================================
UPGRADE 4: ADD RELATIVE TIME EXPLICITLY TO NODE2
==================================================

Add explicit relative time input to the NODE2 vector field.

This should be conceptually consistent with the handbook’s NCDE philosophy, where control includes forcing plus relative time, BUT NODE2 must remain an ODE, not a CDE.

Desired behavior:
- during ODE integration, the NODE2 vector field receives relative time information in addition to latent state, forcing, static embedding, and optional history context
- relative time should be optional
- default enabled

Preferred design:
- normalized relative time over the current rollout interval
- scalar `rel_t` in [0, 1]
- broadcast to nodes and concatenate to the node-wise vector-field input

Requirements:
1. Keep implementation simple and stable.
2. Do not redesign the forcing interpolator.
3. Do not turn NODE2 into a control-path CDE.
4. If the current code already defines solver times explicitly, compute normalized relative time from those solver/query times in a transparent way.
5. Document precisely how relative time is normalized.

New flag:
- `use_relative_time: bool = true`

Optional supporting flag:
- `relative_time_mode: str = "normalized"`

==================================================
API-COMPATIBILITY RULE FOR HISTORYENCODER
==================================================

Preserve the shared `HistoryEncoder` output contract if at all possible:

- `static_embed`
- `step_embeds`
- `hist_context`
- `last_state`
- `last_force`

If existing downstream code expects exact field names or tuple ordering, keep that interface stable unless there is a compelling reason not to.

If old comments/docs refer specifically to `D_lstm`, update wording/comments to something more general like:
- `D_hist`
- `D_context`
but avoid breaking working interfaces unless necessary.

==================================================
IMPLEMENTATION CONSTRAINTS
==================================================

1. Keep all four upgrades optional for ablation.
2. For newly introduced args only, default values should enable the upgraded architecture.
3. For pre-existing args already present in the repo, preserve the repo’s current values.
4. Do not remove the old implementation paths.
5. Do not add new losses or new training objectives.
6. Do not change solver/interpolation/horizon defaults back to handbook defaults if the repo already uses different values.
7. Preserve unrelated config values exactly as they are.
8. Keep code style consistent with the repo.
9. Add clear inline comments for all tensor reshaping and shape semantics.
10. Preserve module boundaries:
    - `HistoryEncoder` as shared front-end
    - `LatentNODEFunc` as NODE2 continuous block
    - `NODE2Model` as assembled model
11. If shared modules must change, ensure NODE1 and NCDE1 still run or remain minimally compatible.
12. Do not implement graph-temporal attention in this upgrade.

==================================================
GRAPH-TEMPORAL ATTENTION CLARIFICATION
==================================================

For documentation purposes, clearly distinguish the implemented design from “graph-temporal attention”.

In this upgrade:
- implemented = temporal-only Transformer after per-step GNN encoding
- NOT implemented = attention jointly across nodes and times, space-time graph transformer, or alternating spatial-attention/temporal-attention blocks

Add a short explanatory note about this distinction inside `upgrade_v1.md`.

==================================================
FILES TO UPDATE
==================================================

Update all relevant files, including but not limited to:

- `models/encoders/history_encoder.py`
- `models/node2_model.py`
- `models/continuous/node_latent_block.py`
- decoder file(s) if separate
- model registry/factory if present
- config loading / argument parsing
- training script(s)
- evaluation script(s) if needed for model construction compatibility
- NODE2-related config files
- shared/base config files if needed for new args
- `.sh` launcher scripts
- documentation

If filenames differ in the actual repo, find the corresponding real files and update them.

==================================================
CONFIG / CLI REQUIREMENTS
==================================================

Wire the new options through the project’s existing config/CLI system.

New args to add (names may be adapted slightly to existing style, but keep them clear and consistent):

- `use_residual_decoder`
- `use_history_in_ode`
- `use_transformer_history`
- `history_encoder_type`
- `history_transformer_num_layers`
- `history_transformer_num_heads`
- `history_transformer_ff_dim`
- `history_transformer_dropout`
- `history_use_positional_encoding`
- optional `history_context_pooling`
- `use_relative_time`
- optional `relative_time_mode`

If the project already prefers YAML config over CLI-only configuration:
- add them to YAML/config
- wire them to CLI if that is already the repo pattern
- do not break existing config override behavior

==================================================
UPDATE .SH SCRIPTS
==================================================

Update all relevant shell scripts so that:
1. the new args are supported
2. the default upgraded mode is explicit where appropriate
3. old ablation-friendly modes remain easy to run

Do not overwrite unrelated experiment-specific values in existing scripts.

==================================================
CREATE / UPDATE upgrade_v1.md
==================================================

Create a detailed markdown file named:

- `upgrade_v1.md`

This file must be substantial and detailed enough for a new lab member.

Required contents:

1. Overview
   - what original NODE2 did
   - what this upgrade adds

2. Original NODE2 vs upgraded NODE2
   Explicitly compare:

   Original:
   - `z0 = InitMLP([hist_context, last_state, static_embed])`
   - `dz/dt = f_theta(z(t), u(t), s, G)`
   - `y_hat(t) = DecMLP(z(t))`

   Upgraded:
   - optional transformer-based `hist_context`
   - optional history-aware ODE evolution
   - optional relative-time input
   - optional residual decoder

3. Upgrade list
   - residual decoder
   - history-aware ODE evolution
   - transformer-based history encoder
   - explicit relative time

4. For each upgrade:
   - motivation
   - mathematical idea
   - implementation details
   - config/CLI args
   - default behavior
   - how to disable for ablation
   - expected benefit
   - possible downside / compute cost

5. Tensor shape walkthrough
   Use the true node-major PyG batched shapes:
   - `x_static`:   [N_total, F_static]
   - `state_hist`: [N_total, H, F_state]
   - `force_hist`: [N_total, H, F_force]
   - `step_embeds`: [N_total, H, D_enc]
   - `hist_context`: [N_total, D_hist]

   Explain clearly:
   - per-step GNN encoding
   - stacking along history dimension
   - temporal-only Transformer over H
   - how `hist_context` is extracted

6. NODE2 forward-path walkthrough
   Step-by-step after the upgrade, including the optional branches.

7. Clarification of what is NOT implemented
   - no graph-temporal attention
   - no horizon-weighted loss
   - no rollout consistency regularization
   - no NODE2-to-NCDE redesign

8. Ablation plan
   Include suggested ablations such as:
   - full upgraded model
   - residual decoder off
   - history-in-ODE off
   - Transformer history off / LSTM history on
   - relative time off
   - all upgrades off = old baseline mode as closely as possible

9. Backward compatibility notes

10. Example commands
   Include example CLI commands for at least:
   a) full upgraded mode
   b) old baseline-like mode
   c) only residual decoder on
   d) Transformer history on but history-in-ODE off
   e) relative time off

11. Existing-config preservation rule
   Explicitly state that pre-existing repo values were preserved and only newly introduced args received new default settings.

==================================================
SELF-CHECK BEFORE FINISHING
==================================================

Before completing the task, verify:

1. all new flags are connected end-to-end
2. default behavior for newly added flags enables the new architecture
3. disabling all new flags restores old behavior as closely as possible
4. existing unrelated config values were preserved
5. scripts still run with the new args
6. `HistoryEncoder` output contract remains stable
7. Transformer shape path is correct under node-major PyG batching
8. relative time is injected correctly and documented
9. residual decoder broadcasts correctly over future steps
10. no new losses/objectives were introduced
11. `upgrade_v1.md` is complete and accurate

If lightweight tests or smoke tests exist, run them.

==================================================
OUTPUT FORMAT AT THE END
==================================================

At the end of your work, provide:

1. a concise summary of what was changed
2. a list of modified files
3. any assumptions made
4. any follow-up issues or TODOs
5. confirmation that existing repo-specific defaults were preserved wherever possible