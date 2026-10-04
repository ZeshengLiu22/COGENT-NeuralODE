# Configuration and evaluation refactor — 2026-10-04

The active workflow now separates training history H, maximum training future K,
and absolute rollout start S. Formal validation, checkpoint selection, testing,
and standalone inference all predict from S to trajectory end. The canonical
entrypoint is `scripts/evaluate.py`; saved complete predictions support later
comparisons without model inference.

## Configuration migration

Previously, dataset YAMLs, duplicated complete history/future bases, combined
architecture presets, and loader overlays mixed data selection, protocol,
architecture, and performance responsibilities. The old complete bases, separate
paper-comparison protocol, combined upgrade presets, and duplicate evaluation
scripts have been removed from the active tree. The shared-anchor restriction
has been deleted entirely; H/K variants have their natural training windows.
No compatibility aliases or filename-dependent protocol branches remain.

The new hierarchy is:

```text
configs/default.yaml
configs/datasets/{issm,anuga,adcirc}.yaml
configs/protocols/{issm,anuga}/main.yaml
configs/models/node2.yaml
configs/ablations/{history,future_len,rollout_start,architecture,temporal_consistency}/
configs/runtime/fast.yaml
```

This is also the merge order, with optional ablations before runtime. Recursive
merging and whole-list replacement are unchanged. Every training entrypoint
writes ordered `config_stack.txt` beside merged `config.json`.

The rollout-start key is now `evaluation.known_steps`. The former validation
mode toggle and evaluation loader settings are removed. Historical names,
old directory structure, and the exact deleted-file inventory are retained only
in the [historical migration details](../old-files/config-evaluation-history-20261004/migration_details.md)
and [complete file inventory](../old-files/config-evaluation-history-20261004/file_inventory.json).
Those files are records of this migration, not active instructions.

## Scientific semantics and preservation

| Protocol | Data | H | K | S | Relative-time scale | Batch / min horizon / loss scale |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ISSM | PIG_5000 | 1 | 180 | 60 | 180 | 8 / 24 / 100 |
| ANUGA | Existing merged simulations | 1 | 64 | 8 | 65 | 1 / 8 / 1 |

At start S, history is `[S-H, ..., S-1]` and prediction covers `[S, ..., T-1]`.
Require `H <= S < T`. A null training horizon maximum still means K. A K30
checkpoint still predicts 180 steps at known60 on T240. Relative-time scale is
fixed for the model and does not change with K or S.

The corrected adapters, normalization, splits, model/ODE implementation,
horizon sampling, and temporal-consistency implementation are unchanged. TC0–5
YAML hyperparameters are preserved. The ADCIRC dataset config explicitly owns
its previously inherited random split. ISSM data selection now lives in its
dataset config; runtime settings contain only loader/cache options.

The default architecture is Transformer + residual + ODE history context +
relative time, latent96. Independent encoder/residual/context/time controls
produce a 16-combination factorial, including LSTM. Explicit default H/K/start,
architecture, and TC0 files remain for experiment readability.

The experiment order is **history → selected-H architecture → selected-model K
→ selected-H/K/architecture TC → inference-only rollout-start robustness**.
Later sweep phases require prior selections. Known60/90/120 use the same final
checkpoint and H; they do not trigger retraining or H reselection.

## Evaluation and artifacts

`Trainer` receives one training loader and three datasets. Validation/testing
read complete trajectories. DDP shards scenarios without padding and reduces
float64 error sums and element counts, including when a rank owns no scenario.
The checkpoint criterion remains `whole_rollout_norm_rmse`.

Standalone evaluation reconstructs the model, normalizer, H/K, and exact split
from the checkpoint. Only explicitly whitelisted runtime values can differ,
including `evaluation.known_steps`; scientific overrides are rejected.
Evaluation output names include split and start to keep each inference distinct.

Schema-v2 compressed NPZ plus JSON metadata preserve complete physical and
normalized predictions/targets, scenario/simulation identity, H/S, absolute
indices, actual adapter times, lead steps, node identity, and mesh/normalization
fingerprints. ANUGA uses supplied simulation time; the preserved ISSM cell
adapter uses snapshot-index time, with no invented physical units.

`scripts/postprocess_rollout.py` provides full recomputation, equal-lead,
common-tail, lead/absolute/actual-time slices, and JSON/CSV reports. It validates
scenario sets, node/channel structure, target/time alignment, normalization, and
requested coverage. Equal-lead120 compares 60–179, 90–209, and 120–239; common
tail compares 120–239 across all starts. Separate initial rollouts are required;
subsequent slices use saved arrays only.

## Important file changes

- Added the layered config tree above; removed all duplicated complete scan
  bases and obsolete combined architecture/config variants.
- Changed `datasets/window_utils.py`, `datasets/base_dataset.py`, and
  `datasets/factory.py` for natural training anchors and trajectory evaluation.
- Changed `training/trainer.py`, `training/evaluator.py`, `scripts/train.py`,
  `scripts/evaluate.py`, and `utils/checkpoint_evaluation.py` for rollout-only
  metrics, checkpoint authority, and config provenance.
- Changed `utils/eval_artifacts.py`; added `utils/rollout_postprocess.py` and
  `scripts/postprocess_rollout.py` for complete artifacts and safe comparisons.
- Updated `train_issm.sh`, `train_anuga.sh`, their NODE2 helpers, and existing
  local/Slurm history/future scheduling helpers. Added architecture/TC sweep
  helpers and `scripts/sweep_config.sh`; removed misleading fixed-H future and
  combined-upgrade wrappers.
- Added `eval_rollout_tmux.sh` and `eval_rollout_start_sweep.sh`; removed duplicate
  inference and obsolete evaluation wrappers.
- Updated all three audit/solver scripts and the dataloader timing experiment.
  Removed the duplicate loader recommendation YAML; the runtime config owns it.
- Added `tests/test_configs.py`, `tests/test_rollout_artifacts.py`, and
  `tests/check_rollout_ddp.py`; replaced obsolete assertions throughout the
  existing configuration, launcher, dataset, checkpoint, training, smoke, and
  TC suites.
- Rewrote `handbook.md` and `config_setting.md`; updated TC documentation and
  regenerated `docs/foundation_audit.json`. Superseded documentation/reports
  moved under `old-files/config-evaluation-history-20261004/` with an explicit
  historical notice. Existing `old-files/` and `legacy-v2/` material stays archived.

The linked file inventory enumerates every added, deleted, moved, and modified
path, including all YAML and shell variants.

## Validation

Validation ran with Python3.11 / PyTorch2.8.0+cu128 in the existing conda
environment, on CPU. CUDA was unavailable. The formal training sweeps were not
launched; the optimization checks use synthetic data and compact models.

- Complete unit/config/launcher/evaluation/artifact/TC/smoke suite: **126 tests
  pass**. Commands and scope are recorded in [validation results](refactor_validation/results.json)
  and the [complete test log](refactor_validation/cpu_tests.txt).
- Two-process Gloo temporal-consistency test: passes rank-specific RNG, dropout
  isolation, global objective reduction, and one forward per training batch.
- Two-process rollout test: passes uneven and empty-rank shards; all metrics
  match single-rank evaluation within 1e-13.
- Real foundation audit: ISSM36 trajectories ×240 snapshots, split28/4/4;
  ANUGA20 ×73, split12/4/4. Canonical values, available horizons, and start bounds
  pass. ANUGA normalized rainfall standard deviation is 0.999999981.
- Real prefix audit: compact untrained model on ISSM2,852 and ANUGA68,464 nodes;
  predictions for four steps exactly match the first four of eight-step runs.
  See [prefix evidence](refactor_validation/real_prefix.json).
- Shell syntax, Python compilation, whitespace checks, and the repository-wide
  active stale-reference audit pass. Obsolete terms remain only in explicitly
  historical directories.
