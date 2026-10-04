# Configuration reference

The [handbook](handbook.md) contains runnable workflows. Configuration is explicit:
`load_config_bundle` loads the files supplied by the caller, in order, without
implicit includes or filename-dependent behavior. `deep_update` recursively
merges dictionaries; scalar values and lists replace earlier values wholesale.

## Ownership and merge order

| Layer | Responsibility |
| --- | --- |
| `configs/default.yaml` | Seed, output, normalization, shared optimizer/horizon curriculum, TC off, evaluation metric/AMP, distributed and safe loader defaults |
| `configs/datasets/<name>.yaml` | Dataset identity, data path, file patterns, split, adapter settings |
| `configs/protocols/<name>/main.yaml` | Dataset-specific H/K, evaluation start, relative-time scale, batch size, minimum training horizon, loss scale |
| `configs/models/node2.yaml` | Encoder, decoder, ODE architecture, solver/interpolation |
| `configs/ablations/...` | One explicit experimental choice |
| `configs/runtime/fast.yaml` | Workers, pinned memory, prefetch, persistent workers, cache |

Use that order. Global defaults do not select a dataset, path, or split. There
are only ISSM and ANUGA main protocols. ADCIRC users supply their intended
protocol settings explicitly. Runtime contains no data choice, split, temporal
length, architecture, or loss settings. Safe defaults already exist in
`default.yaml`; no second runtime-default file is needed.

`config.json` records the merged values and `config_stack.txt` records the exact
ordered paths in every training output directory. CLI curriculum overrides are
reflected in the merged config. Checkpoints embed the merged config, normalizer,
and split manifest for authoritative reconstruction at evaluation.

## The three temporal lengths

All indices below are zero-based. Let T denote the trajectory length.

### History H: `dataset.history_len`

H is the number of true states immediately before the next prediction. For
absolute rollout start S = `evaluation.known_steps`:

```text
start_t = S - 1
history = [S-H, ..., S-1]
future  = [S, ..., T-1]
```

| Configuration | History indices | Future indices for T240 |
| --- | --- | --- |
| H1 / known60 | 59 | 60–239 |
| H6 / known60 | 54–59 | 60–239 |
| H6 / known90 | 84–89 | 90–239 |
| H6 / known120 | 114–119 | 120–239 |

Require `H >= 1`, `S >= H`, and `S < T` for every evaluated trajectory.
H is context length, not an absolute timestep. History cannot be changed at
inference because it belongs to the trained checkpoint.

### Training future K: `dataset.future_len`

K is the maximum future supervision stored in a training sample. A window
anchored at t contains H historical states ending at t and K future states:

```text
history indices = [t-H+1, ..., t]
future indices  = [t+1, ..., t+K]
min_t = H - 1
max_t = T - K - 1
anchors = range(min_t, max_t + 1, dataset.stride)
```

Require positive H, K, and stride. Insufficient trajectory length produces no
training windows. H/K variants naturally have different valid anchors and
window counts. Epoch subsampling remains controlled by
`dataset.sampled_windows.windows_per_scenario` and `epoch_num_windows`.

During training the existing horizon sampler chooses `k_eff <= K` and truncates
future tensors consistently. Keys are:

| Key | Meaning |
| --- | --- |
| `training.train_horizon_mode` | Existing sampler distribution, default `biased_long_horizon` |
| `training.train_horizon_min` | Inclusive lower bound, ISSM 24 / ANUGA 8 |
| `training.train_horizon_max` | Target upper bound; **null resolves to K** |
| `training.train_horizon_curriculum.enabled` | Enable staged upper-bound growth |
| `training.train_horizon_curriculum.epochs` | Curriculum duration, 120 |
| `training.train_horizon_curriculum.warmup_fractions` | Existing stages `[0.40, 0.55, 0.70, 0.85]` |

The existing curriculum/sampling implementation is unchanged. Require
`1 <= train_horizon_min <= target_max <= K`. With K60, ISSM can sample horizons
24–60 once the curriculum reaches its target. Changing K changes the target
maximum automatically when `train_horizon_max` is null.

### Evaluation start S: `evaluation.known_steps`

S is the absolute index of the first predicted state. Evaluation always uses H
true states ending at S−1 and predicts all remaining T−S states. Validation,
final testing, and standalone inference share this rule. No training-window
length controls the formal evaluation horizon:

```text
future_len != formal rollout length
K30 / known60 / T240 -> 180 predicted future states
```

The evaluation section contains `known_steps`, `amp_mode` (default `none`), and
`checkpoint_metric` (default `whole_rollout_norm_rmse`). Checkpoint selection
uses rollout metrics. Training uses only a training-window loader; validation
and test access trajectories directly. DDP distributes whole scenarios by rank
without duplication and reduces exact SSE/absolute-error/count totals.

## Relative-time scale

`model.relative_time_scale` is a fixed property of the trained model, independent
of S and K. Dataset time offsets are relative to the current anchor and measured
in units of the trajectory's median positive timestep. The model uses its saved
scale for its relative-time feature. Existing solver coordinates and time
handling remain unchanged.

ISSM uses scale180 for every H/K and start variant. ANUGA uses scale65 while its
training K remains64. Disabling the relative-time architecture axis disables the
feature; it does not redefine temporal lengths.

```text
relative_time_scale != current rollout length
ISSM known60/90/120 retain scale180, with rollouts180/150/120
ISSM K30/60/120/180 retain scale180
```

## Dataset protocols and preserved model

ISSM uses `./data/ISSM/PIG_5000`, its existing patterns/adapter, and the formal
rate-modulo split: modulo20, validation remainder0, test remainder10. Its main
protocol is H1/K180/known60/scale180, batch8, minimum horizon24, loss scale100.

ANUGA preserves the configured absolute path ending in
`ANUGA/simulation_data_merged`, merged-file patterns, the existing random split,
and corrected adapter behavior. Its main protocol is
H1/K64/known8/scale65, batch1, minimum horizon8, loss scale1. Normalization is
fitted only from training trajectories and restored from the checkpoint.

The default NODE2 model has latent dimension96, Transformer history encoding,
residual decoding, history context in the ODE, and relative time enabled.
Transformer/decoder sizes, activation, interpolation, and solver settings are
preserved in `models/node2.yaml`. Models and data science implementations are
not redesigned by this configuration migration.

## Ablations and selection

History files `h1`–`h8` contain only `dataset.history_len`. Future files contain
only `dataset.future_len`: K30/45/60/75/90/120/150/180 for ISSM, plus K64 as an
ANUGA control. Start files known8/60/90/120 contain only
`evaluation.known_steps` and support inference from a fixed checkpoint.

The independent architecture axes are:

| Directory | Overridden model key | Values |
| --- | --- | --- |
| `architecture/encoder` | `history_encoder.history_encoder_type` | transformer / lstm |
| `architecture/residual` | `use_residual_decoder` | true / false |
| `architecture/ode_context` | `use_history_in_ode` | true / false |
| `architecture/relative_time` | `use_relative_time` | true / false |

The factorial has 16 combinations. Explicit default controls exist for every
axis. Run history first, then architecture with selected H*, then future-length
with selected H*/architecture, then TC with selected H*/K/architecture. The six
TC files retain their previous hyperparameters under
`ablations/temporal_consistency/`; shared defaults disable TC with mode `none`
and weight1. Do not cross TC with the architecture factorial.

Finally run known60/90/120 on the same checkpoint. Keep H* fixed; do not retrain
or reselect H for a new start. Each new start runs inference once, followed by
unlimited model-free slicing of saved results.

## Evaluation authority and output

Standalone evaluation reconstructs model/config/normalizer and split from the
checkpoint. Explicit runtime overrides permit `dataset.data_dir`,
`evaluation.known_steps`, `evaluation.amp_mode`, `output_dir`, and `device`.
Changes to H, K, model, solver, normalization, split, or training values are
rejected. Unchanged scientific values may appear in a supplied config without
altering checkpoint authority. CLI runtime arguments override config overlays.
S must satisfy the checkpoint's H and the evaluated trajectories' lengths.

The saved predictions NPZ and companion metadata JSON retain complete physical
and normalized predictions/targets with scenario/simulation IDs, start/history,
lead steps, absolute timestep indices, adapter-provided time values, and node/mesh identity.
ANUGA saves its supplied simulation time; the existing ISSM cell-format adapter
uses snapshot-index time, which is preserved without inferring physical units.
Post-processing validates alignment and computes RMSE/MAE, per-channel values,
ISSM thickness/speed, final-step metrics, and horizon curves from these arrays.
Equal-lead comparison matches forecast age; common-tail comparison matches
absolute target periods. Arbitrary lead, absolute-index, or physical-time slices
are also supported. Slice bounds and machine-readable JSON/CSV output options
are documented by `python scripts/postprocess_rollout.py --help`.
