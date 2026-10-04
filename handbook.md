# COGENT-NeuralODE operating handbook

Use this guide for the formal ISSM and ANUGA experiments.
The [master technical handbook](docs/continuous_graph_emulator_master_handbook.md)
explains the full data/model pipeline; [config_setting.md](config_setting.md)
defines configuration ownership and [docs/README.md](docs/README.md) indexes
the active technical references.

## Scientific controls

| Quantity | Meaning |
| --- | --- |
| H = `dataset.history_len` | True context states immediately before prediction |
| K = `dataset.future_len` | Maximum stored training future block |
| k_eff ≤ K | Independently sampled actual training prediction length |
| S = `evaluation.known_steps` | Absolute index of first predicted state |
| B = `dataset.train_series_per_scenario_per_epoch` | Selected training series per scenario per epoch |

A training anchor t supplies `x[t-H+1:t+1]` and maximum future
`x[t+1:t+K+1]`. The model predicts the complete first k_eff future states.
Each scenario uses natural legal anchors; B anchors are sampled without
replacement when enough exist. Otherwise every legal anchor appears once,
with replacement only for extra draws needed to reach B, followed by shuffling.

Formal evaluation uses history `x[S-H:S]` and predicts `x[S:T]`.
Require H≤S<T. K does not cap inference: ISSM K30 still predicts 180 steps
from known60 on T240. Checkpoint H/K remain authoritative at inference.

| Protocol | H control | Canonical K | S | B | Train scenarios | Series/epoch | Time scale | Batch/rank | Min k_eff | Loss scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ISSM | 1 | 180 | 60 | 60 | 28 | 1,680 | 180 | 8 | 24 | 100 |
| ANUGA | 1 | 64 | 8 | 9 | 12 | 108 | 65 | 1 | 8 | 1 |

B preserves the canonical H1 natural exposure: 240−1−180+1=60 and
73−1−64+1=9. It does not fix k_eff or total supervised timestep count.
The four-GPU dataset totals require no DDP padding.

Full architecture means Transformer, residual decoder ON, ODE history context
ON, and relative time ON. Time scales stay fixed for all H/K/S variants.

## Configuration and provenance

Merge order is explicit:

```text
default -> dataset -> protocol -> model -> history -> architecture
        -> training_horizon -> rollout_start -> temporal_consistency -> runtime
```

Dictionaries merge recursively; scalars and lists replace earlier values.
Dataset selection/splits live in `configs/datasets/`, scientific baselines in
`configs/protocols/{issm,anuga}/main.yaml`, and model/solver settings in
`configs/models/node2.yaml`. Ablations are separate for each dataset:

```text
configs/ablations/issm/
  history/h1.yaml ... h8.yaml
  architecture/full.yaml
  architecture/a01_transformer_reson_ctxon_timeon.yaml ... a16_lstm_resoff_ctxoff_timeoff.yaml
  training_horizon/k30.yaml ... k180.yaml
  rollout_start/known60.yaml, known90.yaml, known120.yaml
  temporal_consistency/tc0.yaml ... tc5.yaml

configs/ablations/anuga/
  history/h1.yaml ... h8.yaml
  architecture/full.yaml
  architecture/a01_transformer_reson_ctxon_timeon.yaml ... a16_lstm_resoff_ctxoff_timeoff.yaml
  training_horizon/k8.yaml ... k64.yaml
  rollout_start/known8.yaml
  temporal_consistency/tc0.yaml ... tc5.yaml
```

Formal files include all controls even when they match protocol defaults.
Every launcher prints the complete stack including `protocol_config`.
Every training run saves authoritative ordered `config_stack.txt` beside
merged `config.json`. `runtime/fast.yaml` controls DataLoader performance and
does not enable full in-memory caching; the shared default is false.
All formal launchers use `runtime/issm_fast.yaml` or `runtime/anuga_fast.yaml`
to explicitly enable trajectory caching with the same loader settings,
avoiding repeated MAT/NPZ reads and preprocessing for each training series.

## The four formal phases

Only selected history H* propagates from Phase 1. Architecture and K winners
do not determine subsequent phases.

| Phase | H | Architecture | ISSM K / S | ANUGA K / S | TC |
| --- | --- | --- | --- | --- | --- |
| 01_history | 1…8 | Full | 180 / 60 | 64 / 8 | Off |
| 02_architecture | Selected dataset H* | All 16 combinations | 180 / 60 | 64 / 8 | Off |
| 03_training_horizon | Selected dataset H* | Full | K scan / 60 | K scan / 8 | Off |
| 04_temporal_consistency | Selected dataset H* | Full | 180 / 60 | 64 / 8 | TC0…TC5 |

ISSM K scan: 30,45,60,75,90,120,150,180.
ANUGA K scan: 8,16,24,32,40,48,56,64.

Select H* using rollout validation from the history phase. Each later-phase
launcher contains `HISTORY_CONFIG="__SET_SELECTED_HISTORY_AFTER_PHASE1__"`
and exits until edited to the selected dataset-specific history YAML. Update
each file explicitly after selection; no hidden shared selection file supplies
an assumed winner.

## Standalone formal launchers

Four trees contain one complete file for each experiment:

```text
launchers/shell/issm/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/shell/anuga/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/slurm/issm/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/slurm/anuga/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
```

Each tree has 8 history +16 architecture +8 K +6 TC files. Each directly runs
its complete `torchrun` training command with visible config, run name,
process count, interpreter, repository path, and output/log locations. There
are no experiment-enumerating loops, Slurm arrays, common workers, or
dispatcher chains.

From the repository, examples of independent runs are:

```bash
bash launchers/shell/issm/01_history/h1.sh
bash launchers/shell/anuga/01_history/h1.sh

mkdir -p logs
sbatch launchers/slurm/issm/01_history/h1.sh
sbatch launchers/slurm/anuga/01_history/h1.sh
```

Submit each desired experiment's file individually. The shell defaults use
four processes and permit `PROJECT_ROOT`, `PYTHON_BIN`, and `NPROC` overrides.
Formal Slurm files preserve the established H100 resources, four ranks,
24 CPUs per task, nine-hour ISSM/six-hour ANUGA walltime, and cluster Python
environment. Both datasets explicitly charge allocation `TG-CIS250588`.
Inspect the complete file for site details.

Slurm files set the known absolute repository root and `--chdir` explicitly.
They do not infer paths from the submitted script's location or rely on sibling
worker files. Slurm log directories must exist before submission; application
logs and model outputs are under `logs/` and `outputs/<run_name>/`.

`legacy-scripts/` contains archived infrastructure for provenance/reference
only. Do not use those scripts for new formal experiments.

## Ad-hoc entrypoints and direct Python

Top-level `train_issm.sh` and `train_anuga.sh` are direct single-run entrypoints.
They default to `runtime/issm_fast.yaml` and `runtime/anuga_fast.yaml`, respectively,
with full-trajectory caching enabled. Set `RUNTIME_CONFIG=configs/runtime/fast.yaml`
to use the generic loader without caching, or `RUNTIME_CONFIG=''` to omit the
runtime overlay entirely.
Formal launchers do not depend on them. For example:

```bash
bash train_issm.sh
HISTORY_CONFIG=configs/ablations/issm/history/h4.yaml bash train_issm.sh
bash train_anuga.sh
```

They accept named environment overrides for dataset, protocol, model, history,
architecture, training horizon, rollout start, TC, and runtime config. Trailing
arguments are passed directly to `scripts/train.py`. Use these for deliberate
ad-hoc runs, not as selection defaults for formal Phases 2–4.

An explicit single-process canonical ISSM command is:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/datasets/issm.yaml \
  --config configs/protocols/issm/main.yaml \
  --config configs/models/node2.yaml \
  --config configs/ablations/issm/history/h1.yaml \
  --config configs/ablations/issm/architecture/full.yaml \
  --config configs/ablations/issm/training_horizon/k180.yaml \
  --config configs/ablations/issm/rollout_start/known60.yaml \
  --config configs/ablations/issm/temporal_consistency/tc0.yaml \
  --config configs/runtime/issm_fast.yaml \
  --run-name issm_h1_example
```

Run using the intended PyTorch environment with `requirements.txt` installed.
ADCIRC has an adapter/config, but no supplied formal main protocol.

## Training outputs and evaluation

Training saves `config.json`, `config_stack.txt`, `split_files.json`,
`train.log`, `history.json`, `best.pt`, and `final_metrics.json`.
Validation selects `whole_rollout_norm_rmse`. Test uses the best checkpoint.
The checkpoint embeds config, normalizer, and exact split membership.

`scripts/evaluate.py` is the formal standalone evaluation entrypoint:

```bash
python scripts/evaluate.py --checkpoint outputs/final/best.pt --split test \
  --config configs/ablations/issm/rollout_start/known60.yaml \
  --output-dir outputs/rollouts
```

`--known-steps` can set the start directly. Runtime overrides may relocate
`dataset.data_dir` or select evaluation AMP/device/output; checkpoint H, K,
architecture, solver, normalization, training settings, and splits remain
authoritative. Validation/test/evaluation datasets construct no training
anchors and access scenarios through rollout data.

For ISSM robustness, evaluate the same checkpoint from known60/90/120:

```bash
bash eval_rollout_start_sweep.sh --checkpoint outputs/final/best.pt \
  --output-dir outputs/rollouts
```

This helper contains three explicit evaluation calls and does not train or
reselect a model. Each start reanchors true history, producing 180/150/120
future steps for T240. ANUGA retains known8. `eval_rollout_tmux.sh` can launch
one evaluation in a detached tmux session.

Complete compressed NPZ predictions and companion metadata JSON use the stem
`best.<split>.known<S>.full_rollout`. They retain every future node/channel,
physical and normalized predictions/targets, scenario/simulation identities,
H/S, lead steps, absolute indices, adapter time, and mesh/normalizer identity.
`--plot-max-lead-step` limits plots only, leaving full arrays and metrics.
ANUGA can additionally export time series and flood maps.

## Postprocessing without inference

```bash
python scripts/postprocess_rollout.py \
  --artifacts outputs/rollouts/best.test.known{60,90,120}.full_rollout_predictions.npz \
  --mode equal-lead --lead-steps 120 --output-prefix outputs/rollouts/equal120
python scripts/postprocess_rollout.py \
  --artifacts outputs/rollouts/best.test.known{60,90,120}.full_rollout_predictions.npz \
  --mode common-tail --output-prefix outputs/rollouts/common_tail
```

Equal lead compares forecast age: absolute periods 60–179/90–209/120–239.
Common tail compares the same target period 120–239 across starts. The
postprocessor validates alignment and writes JSON/CSV without loading a model.
Other modes include `full`, `lead-slice`, `absolute-slice`, and `time-slice`;
`--help` documents inclusive/exclusive bounds. Keep each NPZ and its metadata
together.

## Validation and pre-sweep CUDA smoke

The CPU/unit and real-data audit commands are:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 python tests/check_temporal_ddp.py
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 python tests/check_rollout_ddp.py
python -m compileall .
python scripts/audit_foundation.py --data-root data --output docs/foundation_audit.json
python scripts/audit_relative_time.py --data-root data --output docs/final_cleanup_validation/real_prefix.json
```

Syntax-check active shell/Slurm files and exclude historical directories
`legacy-scripts/` and `old-files/` from active stale-reference
audits. [Current validation evidence](docs/final_cleanup_validation/results.json)
records completed checks; older `docs/refactor_validation/` is a historical
snapshot of the earlier refactor.

Before the large sweeps, run the separate bounded CUDA smoke jobs from an
authenticated cluster login:

```bash
mkdir -p logs
sbatch tests/smoke_cuda_issm.sh
sbatch tests/smoke_cuda_anuga.sh
```

These are validation jobs, not formal results. They exercise real canonical
data/configuration, fixed B, four-rank training with sampled k_eff, rollout
validation, best checkpoint, standalone evaluation, complete NPZ, and
postprocessing. Their bounded optimizer run does not constitute a formal
epoch or accuracy result. CUDA training is validated only when the actual jobs
complete and their evidence is inspected; CPU success alone does not establish it.
