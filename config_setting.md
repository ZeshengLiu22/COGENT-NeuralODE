# Configuration reference

[handbook.md](handbook.md) gives operating commands. `load_config_bundle` loads
only files explicitly supplied by the caller, in order. `deep_update` merges
dictionaries recursively; scalars and lists replace earlier values wholesale.
Filenames do not select behavior.

## Layer ownership and order

| Layer, in merge order | Responsibility |
| --- | --- |
| `configs/default.yaml` | Shared seed/output, loader defaults with trajectory caching enabled, normalization metadata, optimizer, horizon sampling/curriculum, TC off, evaluation metric/AMP, distributed settings |
| `configs/datasets/<dataset>.yaml` | Dataset identity, path, file patterns, adapter, split |
| `configs/protocols/<dataset>/main.yaml` | H/K/S, per-scenario B, fixed time scale, batch size, minimum horizon, loss scale |
| `configs/models/node2.yaml` | NODE2 dimensions, architecture defaults, solver/interpolation |
| `configs/ablations/<dataset>/history/hN.yaml` | Explicit history H |
| `configs/ablations/<dataset>/architecture/*.yaml` | All four architecture choices |
| `configs/ablations/<dataset>/training_horizon/kN.yaml` | Maximum training future K |
| `configs/ablations/<dataset>/rollout_start/knownN.yaml` | Evaluation start S |
| `configs/ablations/<dataset>/temporal_consistency/tcN.yaml` | TC0–TC5 |
| `configs/runtime/{issm,anuga}_fast.yaml` | Formal dataset-specific workers, pinned memory, prefetching, persistent-worker request |
| `configs/runtime/single_a100.yaml` | Optional single-A100 override: `training.grad_accum_steps: 4`; unused by the formal launchers |
| `configs/runtime/issm_derecho.yaml` | ISSM on Derecho final override: per-rank `training.batch_size: 4`, `training.grad_accum_steps: 2` |

Each formal stack includes control overlays even when they match protocol
defaults. Dataset-scoped ISSM and ANUGA files are intentionally explicit and
duplicated. No shared architecture/TC tree hides formal provenance.

The shared default is `dataset.cache_in_memory: true`, reusing full trajectories
across series and epochs, including runs with no runtime overlay.
Formal ISSM/ANUGA launchers and the top-level `train_issm.sh`/`train_anuga.sh`
defaults select `runtime/issm_fast.yaml` or `runtime/anuga_fast.yaml` for four
workers, pinned memory, prefetching, and persistent-worker requests. These
loader overlays inherit trajectory caching from `configs/default.yaml`.
Epoch resampling requires workers to see current series indices, so training
disables persistent workers when needed despite the runtime request.

Shared `amp.mode` and `evaluation.amp_mode` are both `none`, disabling AMP
for shell, Slurm, and PBS. All formal launchers use four ranks by default:

| Launchers | Dataset | Batch per rank | Accumulation | Effective global batch |
| --- | --- | --- | --- | --- |
| shell / Slurm | ISSM | 8 | 1 | 32 |
| Derecho PBS | ISSM | 4 | 2 | 32 |
| shell / Slurm / Derecho PBS | ANUGA | 1 | 1 | 4 |

Every ISSM PBS launcher appends `configs/runtime/issm_derecho.yaml` after all
other configs. It changes only batch size and accumulation; all selected H/K/S,
architecture, TC, learning-rate, scheduler, epoch, and series-budget settings
come from the preceding layers. ANUGA PBS uses its protocol batch size and
shared accumulation default. For ad-hoc ISSM runs on Derecho, append the same
override explicitly. `single_a100.yaml` remains available for optional
single-GPU use and is not selected by the formal launchers.

`config_stack.txt` records ordered supplied paths; `config.json` records
merged values including the optional CLI curriculum override. Checkpoints
embed merged configuration, normalizer, and split manifest.

## Temporal quantities and the series budget

All indices are zero-based, slice stops exclusive.

| Symbol | Setting | Meaning |
| --- | --- | --- |
| H | `dataset.history_len` | True historical states ending at the anchor |
| K | `dataset.future_len` | Maximum stored training future |
| S | `evaluation.known_steps` | Absolute first predicted state in rollout evaluation |
| B | `dataset.train_series_per_scenario_per_epoch` | Selected training series per simulation per epoch |
| $k_{\mathrm{eff}}$ | Sampled in trainer | Actual complete-series training prediction length |

At training anchor t:

$$
\begin{aligned}
\mathrm{history} &= x[t-H+1:t+1],\\
\mathrm{maximum\ future} &= x[t+1:t+K+1],\\
\mathrm{natural\ anchors} &= H-1,\ldots,T-K-1,\\
N_{\mathrm{natural}} &= T-H-K+1
\quad\text{(stride 1 and }T\ge H+K\text{)}.
\end{aligned}
$$

H, K, and stride must be positive. Natural anchors vary with H/K. B is a
positive integer in the formal protocols; null retains all natural anchors
for an explicitly ad-hoc configuration. There are no retired sampling-key
aliases. A positive B with no legal anchors is an error.

For $N\ge B$, select exactly B distinct anchors without replacement. For $0<N<B$,
include all N once, sample $B-N$ extras with replacement from that same scenario,
and shuffle its B entries. Seed plus epoch determines selection; horizons
are sampled independently later.

One selected training series is identified by a simulation/scenario and a
natural history anchor. Its actual prediction length is determined later by
the independently sampled $k_{\mathrm{eff}}$. Fixing B does not fix unique-anchor count,
target-timestep count, ending time, or $k_{\mathrm{eff}}$.

| Protocol | T | Canonical H/K/S | B | Train scenarios | Series per epoch |
| --- | ---: | --- | ---: | ---: | ---: |
| ISSM | 240 | 1/180/60 | 60 | 28 | 1,680 |
| ANUGA | 73 | 1/64/8 | 9 | 12 | 108 |

Budgets derive from canonical natural counts: $240-1-180+1=60$ and
$73-1-64+1=9$. For ANUGA H4/K64, legal anchors 3–8 give six unique plus three
extra draws. H8/K64 gives anchors 7–8 once each plus seven extras.
All formal H/K variants keep the same per-dataset epoch length, divisible
by four for the current DDP runs.

Training builds natural-anchor metadata; val/test/standalone evaluation does
not. Evaluation still supports scenario metadata, trajectory iteration,
normalization, and rollout construction. Dataset length for these datasets
is zero training series; use `len(scenario_infos)` for scenario count.

## $k_{\mathrm{eff}}$ and curriculum

| Setting | Current meaning |
| --- | --- |
| `training.train_horizon_mode` | `biased_long_horizon` default; `uniform_random` also supported |
| `training.train_horizon_min` | Inclusive minimum 24 ISSM / 8 ANUGA |
| `training.train_horizon_max` | Target maximum; null inherits K |
| `training.train_horizon_curriculum.enabled` | Enables epoch staircase |
| `training.train_horizon_curriculum.epochs` | Default 120 epochs |
| `training.train_horizon_curriculum.warmup_fractions` | Default [0.40,0.55,0.70,0.85] |

Require $1\le\mathrm{minimum}\le\mathrm{target}\le K$. ANUGA K8 is valid with minimum 8.
The epoch cap uses the listed staircase fractions, round-half-up, and clamping;
after the curriculum it equals the target. See [exact equations](docs/upgrade_v1.1.md).

Rank 0 samples $k_{\mathrm{eff}}$ once per optimizer/accumulation group and broadcasts it.
The trainer truncates forcing, targets, times, and future metadata, then NODE2
predicts every state $1,\ldots,k_{\mathrm{eff}}$. B sampling does not change this logic. Optimizer
and cosine scheduler remain epoch-based.

## Rollout start and time scale

Formal history is `x[S-H:S]` and prediction is `x[S:T]`, requiring $H\le S<T$.
Validation, best-checkpoint selection, final test, and standalone inference
share this rule. K30/known60/T240 predicts 180 future states; K does not cap
rollout evaluation.

| Example | History indices | Future indices for T240 |
| --- | --- | --- |
| H1/known60 | 59 | 60–239 |
| H6/known60 | 54–59 | 60–239 |
| H6/known90 | 84–89 | 90–239 |
| H6/known120 | 114–119 | 120–239 |

Solver time is relative to the current anchor, divided by the trajectory's
median positive timestep. The optional vector-field feature is
$r(t)=t/\texttt{model.relative\_time\_scale}$ with fixed scale 180 ISSM / 65 ANUGA.
That scale is independent of H, K, S, and the inference endpoint. It is not
clamped. Disabling the feature is one architecture axis, not a time-unit change.

`evaluation.amp_mode` defaults to `none`, and the checkpoint metric is
`whole_rollout_norm_rmse`. DDP evaluation assigns complete scenarios without
padding and reduces float64 error sums and element counts.

## Dataset and model controls

ISSM uses `./data/ISSM/PIG_5000` and rate-modulo splitting: modulo 20, validation
remainder 0, test remainder 10. The main protocol has batch 8, minimum horizon 24,
and loss scale 100.

ANUGA defaults to `./data/ANUGA/simulation_data_merged` in every launcher,
relative to `PROJECT_ROOT`, with the merged-file pattern and 0.6/0.2/0.2 random
split seeded at 42. It has batch 1, minimum horizon 8, and loss scale 1.
Normalization uses training trajectories only and is restored from checkpoints.
Derecho PBS runs ANUGA on four GPUs with accumulation 1.

ADCIRC has a dataset configuration but no supplied formal main protocol.
Ad-hoc ADCIRC use requires explicit temporal/training settings.

Canonical NODE2 has width 96, Transformer history, residual decoder, persistent
history context in the ODE, and relative time. Every architecture YAML explicitly
sets these keys:

```yaml
model:
  history_encoder:
    history_encoder_type: transformer  # or lstm
  use_residual_decoder: true           # or false
  use_history_in_ode: true             # or false
  use_relative_time: true              # or false
```

`full.yaml` equals `a01_transformer_reson_ctxon_timeon.yaml`; a01–a16 cover the
complete four-factor Cartesian product. [Architecture equations](docs/upgrade_v1.md)
explain what each switch changes. Model dimensions, interpolation, solver, and
fixed protocol time scales stay intact across the factorial.

## Formal ablations and selection

History files h1–h8 change H. ISSM K files are
30/45/60/75/90/120/150/180; ANUGA K files are 8/16/24/32/40/48/56/64.
Rollout-start files are ISSM known60/90/120 and ANUGA known8.
TC files tc0–tc5 preserve the corrected implementation documented in
[temporal consistency](docs/temporal_consistency.md).

1. History: full architecture, canonical K/S, TC0; select dataset-specific $H^*$.
2. Architecture: selected $H^*$, canonical K/S, TC0; all 16 combinations.
3. K: selected $H^*$, full architecture, canonical S, TC0; dataset K scan.
4. TC: selected $H^*$, full architecture, canonical K/S; TC0–TC5.

Only $H^*$ propagates. The architecture and K winners do not define later phases.
All later-phase standalone launcher files require explicit selected-history
edits after Phase 1; no H is guessed. ISSM start robustness is evaluation-only
using one final checkpoint, with no retraining/reselection.

## Standalone evaluation authority and artifacts

Evaluation reconstructs the checkpoint config, model, normalizer, and exact
split manifest. Permitted runtime changes are `dataset.data_dir`,
`evaluation.known_steps`, `evaluation.amp_mode`, `output_dir`, and `device`.
H, K, model, solver, normalization, split, and training changes are rejected.
Unchanged scientific values can appear in supplied YAML without changing
checkpoint authority. CLI runtime arguments override runtime overlays.

Complete predictions NPZ and companion metadata preserve physical/normalized
predictions and targets, scenario/simulation IDs, H/S, absolute indices, lead
steps, adapter times, and node/mesh/normalizer identity. ANUGA times preserve
its supplied coordinates; ISSM cell-format times are snapshot indices.
Postprocessing validates alignment and supports equal-lead, common-tail, and
explicit lead/index/time slices without model inference. Use
`python scripts/postprocess_rollout.py --help` for bounds and JSON/CSV output.
