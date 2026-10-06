# Configuration, rollout evaluation, and final experiment cleanup — 2026-10-04

The active design separates H (history), K (maximum stored training future),
S (rollout start), and B (per-scenario epoch series count). Formal validation,
checkpoint selection, test, and standalone inference predict from S to the
trajectory end. Only selected history $H^*$ propagates between formal phases.

This record incorporates the final cleanup after the earlier rollout-only
refactor. [Current validation evidence](final_cleanup_validation/results.json)
records the final checks. Reports under `refactor_validation/` describe the
earlier commit and are not evidence for subsequent changes.

## History reviewed and reconciled

The implementation/documentation review covered these milestones:

| Commit | Useful historical context | Current disposition |
| --- | --- | --- |
| `d28666f` | Natural legal-anchor enumeration; original master and architecture/curriculum explanations | Preserve natural H/K-specific anchors and recover useful technical explanations |
| `22e5298` | ANUGA history scans and complete rollout exports | Preserve canonical ANUGA forecasting and full artifacts |
| `961c3eb` | Foundation repair: ISSM inputs, normalization, fixed relative time, checkpoint provenance; earlier common-anchor budgets | Preserve correctness repairs; replace common-anchor budgeting with B |
| `f96cae3` | Configurable TC objectives | Preserve state plus optional TC |
| `f8beea4` | Corrected per-graph pair sampling/RNG, stable RMSE, and launcher protocol preservation | Retain corrected loss behavior and cluster settings |
| `b3c761b` | H/K/S config separation, natural anchors, rollout-only evaluation, complete postprocessing artifacts | Retain scientific meanings and extend with controlled epoch exposure |

Historical explanations were checked against current adapters, normalizer,
encoders, ODE, trainer, losses, evaluator, and artifact code. Archived concepts
are not restored as runtime behavior. The active technical documents are in
[the docs index](README.md); historical material under `old-files/` and
`legacy-scripts/` is provenance only.

## Fixed exposure with natural anchors and variable horizons

At anchor t, a training sample holds history `x[t-H+1:t+1]` and maximum future
`x[t+1:t+K+1]`. Its actual prediction is the complete first $k_{\mathrm{eff}}$ future states,
with $k_{\mathrm{eff}}$ sampled independently by the existing synchronized horizon sampler.

At stride 1,

$$
N_{\mathrm{natural}}=T-H-K+1.
$$

Each scenario contributes B selected series. If $N\ge B$, choose B distinct anchors
without replacement. If $0<N<B$, include all anchors once, draw $B-N$ extras with
replacement from that same scenario, then shuffle. This preserves unique-anchor
coverage when the budget exceeds the natural set. A positive budget with no
legal anchors is an error.

`dataset.train_series_per_scenario_per_epoch` is the only epoch sampling
budget field. Formal ISSM $B=60$ preserves canonical H1/K180 exposure on T240;
ANUGA $B=9$ preserves H1/K64 exposure on T73. Therefore $28\times60=1{,}680$ ISSM and
$12\times9=108$ ANUGA training series are selected each epoch across every formal H/K
variant. Both counts divide evenly among four ranks, requiring no DDP padding.

This equalizes scenario exposure and series count, not $k_{\mathrm{eff}}$, total target
timesteps, exact anchors, or series ending time. The epoch-based optimizer/
scheduler behavior is retained. The natural anchor set remains specific to H/K.
For ANUGA H4/K64, six legal anchors yield six unique plus three repeated draws;
H8/K64 yields both legal anchors once plus seven repeated draws.

Validation/test/evaluation datasets no longer enumerate training anchors.
They retain full trajectory access, normalizer, scenario identity, and mesh
metadata. Standalone train-split evaluation also skips training-series expansion.

## Scientific protocol preservation

| Protocol | Data | Canonical H/K/S | B | Relative-time scale | Batch / minimum horizon / loss scale |
| --- | --- | --- | ---: | ---: | --- |
| ISSM | PIG_5000 | 1 / 180 / 60 | 60 | 180 | 8 / 24 / 100 |
| ANUGA | Existing merged simulations | 1 / 64 / 8 | 9 | 65 | 1 / 8 / 1 |

Formal history is `x[S-H:S]`, prediction `x[S:T]`, and $H\le S<T$.
K30 does not cap an ISSM known60 rollout: it still predicts 180 states on T240.
Null `train_horizon_max` inherits K; ANUGA K8 is valid with minimum 8.
The relative-time scale remains fixed independently of K, S, or endpoint.

The corrected ISSM inputs, ANUGA rainfall units/normalization, scenario splits,
NODE2 dynamics, full-series loss, horizon curriculum, and corrected TC behavior
are preserved. H/K remain checkpoint-authoritative in standalone evaluation;
`evaluation.known_steps` remains the allowed temporal override.

## Explicit dataset-scoped configurations and phases

The stack is:

```text
default -> dataset -> protocol -> model -> history -> architecture
        -> training_horizon -> rollout_start -> temporal_consistency -> runtime
```

Scientific overlays live in separate `configs/ablations/issm/` and
`configs/ablations/anuga/` trees. Each has history, architecture, training_horizon,
rollout_start, and temporal_consistency directories. Each architecture file
explicitly sets encoder, residual, ODE context, and relative time. `full.yaml`
matches a01; a01–a16 cover every four-factor combination. TC overlays are
`tc0.yaml` through `tc5.yaml`. The generic `runtime/fast.yaml` keeps caching off;
formal launchers use `runtime/issm_fast.yaml` or `runtime/anuga_fast.yaml` to
explicitly enable full-trajectory caching for these datasets.

| Phase | H | Architecture | K | S | TC |
| --- | --- | --- | --- | --- | --- |
| 1. History | 1…8 | Full | Canonical | Canonical | Off |
| 2. Architecture | Selected $H^*$ | All 16 | Canonical | Canonical | Off |
| 3. K | Selected $H^*$ | Full | Dataset K scan | Canonical | Off |
| 4. TC | Selected $H^*$ | Full | Canonical | Canonical | TC0…TC5 |

ISSM K scan is 30/45/60/75/90/120/150/180; ANUGA is
8/16/24/32/40/48/56/64. Architecture and K winners do not propagate to other
phases. Later-phase scripts contain visible fail-fast selected-H placeholders.

ISSM known60/90/120 robustness is separate evaluation-only work using the same
final checkpoint without retraining/reselection. ANUGA retains known8.

## Launcher and documentation changes

Formal launchers occupy four trees under
`launchers/{shell,slurm}/{issm,anuga}/` with phase directories 01–04.
Each tree contains 38 complete independent experiment files: 8 history,
16 architecture, 8 K, 6 TC. There are no experiment loops, Slurm arrays, generic
workers, or shared experiment dispatchers. Slurm files use explicit known
repository roots and copied environment/resource settings, avoiding spooled
script path inference.

Top-level training scripts remain direct ad-hoc single-run entrypoints.
Every active training launcher prints the full config stack including protocol.
`config_stack.txt` remains the saved authoritative order. Historical launcher
directories are retained beneath `legacy-scripts/`, excluded from active
launcher/stale-reference audits, and are not recommended for new runs.

Restored active references are the substantial master technical handbook,
architecture equations, exact curriculum equations, and formula-complete TC
guide. The operating/configuration guides and docs index point to current
paths and $H^*$-only phase dependencies.

## Evaluation artifacts and postprocessing

Evaluation reconstructs the checkpoint model, normalizer, H/K, solver, and exact
splits. Runtime-only overrides allow data relocation, S, precision, output
location, and device. DDP evaluates scenario shards without padding and reduces
float64 error sums/counts. Selection remains `whole_rollout_norm_rmse`.

Complete schema-v2 compressed NPZ plus JSON metadata preserve normalized/
physical predictions and targets, scenario/simulation identity, H/S, absolute
indices, adapter times, lead steps, node identity, and mesh/normalizer
fingerprints. ANUGA preserves supplied time; ISSM cell-format data preserve
snapshot-index time without invented physical units.

Postprocessing recomputes metrics without a model, validates artifact alignment,
and supports equal-lead, common-tail, and lead/index/time slices. For ISSM
starts 60/90/120, equal-lead 120 compares periods 60–179/90–209/120–239;
common-tail compares 120–239 for every start.

## Validation scope

The current [results record](final_cleanup_validation/results.json) and its
linked logs are the evidence for CPU/unit checks, two-process Gloo TC and
rollout checks, Python compilation, active shell syntax, and real-data
foundation/prefix audits. The foundation audit exercises the actual series
sampler across formal H/K scans, and the prefix audit uses rollout datasets.

A bounded real-data CUDA smoke runner and independent ISSM/ANUGA submission
files are provided under `tests/`. They exercise canonical model/data/splits,
fixed series counts, four-rank training, sampled $k_{\mathrm{eff}}$, rollout validation,
best checkpoint, standalone evaluation, saved NPZ, and postprocessing.
CUDA smoke has not been executed as of this cleanup record: this session's
compute-node submission policy and login authentication prevent submission.
Run the two smoke scripts from an authenticated cluster login and inspect their
results before large sweeps. Neither CPU validation nor a smoke run is a formal
accuracy result; no formal sweep results are claimed here.
