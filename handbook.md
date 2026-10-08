# COGENT-NeuralODE operating handbook

Use this guide for the formal ISSM and ANUGA experiments.
The [master technical handbook](docs/continuous_graph_emulator_master_handbook.md)
explains the full data/model pipeline; [config_setting.md](config_setting.md)
defines configuration ownership and [docs/README.md](docs/README.md) indexes
the active technical references.

## Scientific controls

| Quantity | Meaning |
| --- | --- |
| $H=\texttt{dataset.history\_len}$ | True context states immediately before prediction |
| $K=\texttt{dataset.future\_len}$ | Maximum stored training future block |
| $k_{\mathrm{eff}}\le K$ | Independently sampled actual training prediction length |
| $S=\texttt{evaluation.known\_steps}$ | Absolute index of first predicted state |
| $B=\texttt{dataset.train\_series\_per\_scenario\_per\_epoch}$ | Selected training series per scenario per epoch |

A training anchor t supplies `x[t-H+1:t+1]` and maximum future
`x[t+1:t+K+1]`. The model predicts the complete first $k_{\mathrm{eff}}$ future states.
Each scenario uses natural legal anchors; B anchors are sampled without
replacement when enough exist. Otherwise every legal anchor appears once,
with replacement only for extra draws needed to reach B, followed by shuffling.

Formal evaluation uses history `x[S-H:S]` and predicts `x[S:T]`.
Require $H\le S<T$. K does not cap inference: ISSM K30 still predicts 180 steps
from known60 on T240. Checkpoint H/K remain authoritative at inference.

| Protocol | H control | Canonical K | S | B | Train scenarios | Series/epoch | Time scale | Batch/rank | Min $k_{\mathrm{eff}}$ | Loss scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ISSM | 1 | 180 | 60 | 60 | 28 | 1,680 | 180 | 8 | 24 | 100 |
| ANUGA | 1 | 64 | 8 | 9 | 12 | 108 | 65 | 1 | 8 | 1 |

B preserves the canonical H1 natural exposure: $240-1-180+1=60$ and
$73-1-64+1=9$. It does not fix $k_{\mathrm{eff}}$ or total supervised timestep count.
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
merged `config.json`. Full in-memory trajectory caching is enabled in
`configs/default.yaml`, avoiding repeated MAT/NPZ reads and preprocessing for
each training series. All formal launchers use `runtime/issm_fast.yaml` or
`runtime/anuga_fast.yaml` for worker, pinned-memory, and prefetch settings.
Shared training AMP (`amp.mode`) and evaluation AMP (`evaluation.amp_mode`)
default to `none` for shell, Slurm, and PBS runs. ISSM PBS appends
`runtime/issm_derecho.yaml` last for batch size 4 and accumulation 2.
ANUGA PBS uses batch size 1 and accumulation 1; both launch four ranks.

## The four formal phases

Only selected history $H^*$ propagates from Phase 1. Architecture and K winners
do not determine subsequent phases.

| Phase | H | Architecture | ISSM K / S | ANUGA K / S | TC |
| --- | --- | --- | --- | --- | --- |
| 01_history | 1…8 | Full | 180 / 60 | 64 / 8 | Off |
| 02_architecture | Selected dataset $H^*$ | All 16 combinations | 180 / 60 | 64 / 8 | Off |
| 03_training_horizon | Selected dataset $H^*$ | Full | K scan / 60 | K scan / 8 | Off |
| 04_temporal_consistency | Selected dataset $H^*$ | Full | 180 / 60 | 64 / 8 | TC0…TC5 |

ISSM K scan: 30,45,60,75,90,120,150,180.
ANUGA K scan: 8,16,24,32,40,48,56,64.

Select $H^*$ using rollout validation from the history phase. Each later-phase
launcher contains `HISTORY_CONFIG="__SET_SELECTED_HISTORY_AFTER_PHASE1__"`
and exits until edited to the selected dataset-specific history YAML. Update
each file explicitly after selection; no hidden shared selection file supplies
an assumed winner.

## Standalone formal launchers

Six trees contain one complete file for each experiment:

```text
launchers/shell/issm/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/shell/anuga/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/slurm/issm/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/slurm/anuga/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/PBS/issm/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
launchers/PBS/anuga/{01_history,02_architecture,03_training_horizon,04_temporal_consistency}/
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
24 CPUs per task, a 12-hour walltime for both datasets, and cluster Python
environment. Both datasets explicitly charge allocation `TG-CIS250588`.
Inspect the complete file for site details.

Slurm files set the known absolute repository root and `--chdir` explicitly.
They do not infer paths from the submitted script's location or rely on sibling
worker files. Slurm log directories must exist before submission; application
logs and model outputs are under `logs/` and `outputs/<run_name>/`.

PBS files mirror all 76 shell/Slurm experiments on Derecho. Each requests
one exclusive GPU node in `main` under `ULHI0006`, with 64 CPUs, four A100 40GB
GPUs, and `walltime=12:00:00`. The select request is
`select=1:ncpus=64:mpiprocs=4:ompthreads=1:ngpus=4`, with `place=excl`;
host memory uses the GPU queue's full-node default (currently 487GB).
Job names identify the dataset, phase, and variant (e.g. `issm_01_h1`).
They load the `cuda` module and use `NPROC=4` with `torchrun --standalone`.
ISSM appends `configs/runtime/issm_derecho.yaml` last for per-rank batch size 4
and accumulation 2, giving effective global batch 32. ANUGA uses batch size 1
and accumulation 1, giving effective global batch 4. Each experiment retains
its selected K, H, S, architecture, TC, learning rate, scheduler, and 300 epochs.

PBS defaults to `/glade/u/home/zel/scratch/COGENT-NeuralODE` and
`/glade/work/zel/conda-envs/derecho-ml/bin/python`; `PROJECT_ROOT` and
`PYTHON_BIN` can be passed with `qsub -v` to override these paths. The project
root is explicit so scheduler spooling does not affect path resolution.
Both dataset configurations use paths relative to `PROJECT_ROOT`: ISSM uses
`./data/ISSM/PIG_5000`, and ANUGA uses `./data/ANUGA/simulation_data_merged`
for shell, Slurm, and PBS runs.

```bash
cd /glade/u/home/zel/scratch/COGENT-NeuralODE
qsub launchers/PBS/issm/01_history/h1.sh
qsub launchers/PBS/anuga/01_history/h1.sh
```

Every PBS script sets `#PBS -m abe` and `#PBS -M zel220@lehigh.edu`.
The scheduler emails that address when the job begins, ends, or is aborted,
including training/inference failures and walltime termination. To choose a
different recipient for a submission, use `qsub -M address@example.com <script>`.
These are the standard [PBS mail directives](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/#other-frequently-used-pbs-directives).

PBS joins scheduler stdout/stderr (`#PBS -j oe`). Phases 2–4 still require
selecting history in each file. Each submission runs training to completion,
checks that `train/best.pt` is a nonempty regular file, then invokes the existing
evaluation entrypoint once:

```bash
"$PYTHON_BIN" scripts/evaluate.py \
  --checkpoint "$TRAIN_DIR/best.pt" \
  --split test --device cuda --amp-mode none \
  --output-dir "$INFER_DIR"
```

The experiment name retains dataset, phase, H, K, architecture, and TC, followed
by a UTC timestamp with nanoseconds and the PBS job ID (local PID outside PBS).
The launcher reserves the run folder atomically and refuses to reuse one that
already exists. Training uses `--run-name "$RUN_NAME/train"`; inference shares
the same parent folder:

```text
outputs/<experiment>_<timestamp>_pbs<job_id>/
  train/
    best.pt
    config.json
    config_stack.txt
    split_files.json
    history.json
    final_metrics.json
    train.log
    launcher.log
    runtime_metadata.txt
  inference/
    inference.log
    metric_table.txt
    best.test.known<S>.full_rollout_metrics.json
    best.test.known<S>.full_rollout_metric_table.csv
    best.test.known<S>.full_rollout_summary_table.csv
    best.test.known<S>.full_rollout_predictions.npz
    best.test.known<S>.full_rollout_predictions_meta.json
    ... existing curve CSV/NPZ files and plots
    ... ANUGA flood-map directory and complete per-scenario time series
```

Evaluation restores the best checkpoint's split manifest, normalizer, history,
and known steps. ISSM defaults to S=60 and predicts steps 61–240; ANUGA defaults
to S=8 and predicts through the trajectory end. K never truncates this rollout.
The training process's existing final TEST metrics remain in `train/`;
standalone inference additionally exports predictions and analysis artifacts.

`metric_table.txt` contains exactly the existing `format_metric_table` output
plus a final newline. The same complete table appears under `Detailed metric
table:` in `inference.log`, with whole-rollout/final-step rows for aggregate
and individual channels. Existing CSV tables, metric definitions, units,
precision, rounding, plots, and ANUGA exports are preserved.

Training failure skips inference. Missing/empty `best.pt` fails explicitly.
Inference failure returns a nonzero job status and retains the completed
training outputs and inference error log. No cleanup removes partial runs.
Best-effort metadata records the job ID, Git commit, GPU model, experiment,
configuration stack, stage start/end UTC timestamps, and exit statuses in
`train/runtime_metadata.txt` and `train/launcher.log`. Metadata failures do not
block the experiment. PBS still manages its own joined scheduler output.

The [NCAR Derecho queue documentation](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/charging/#derecho-queues)
sets a 12-hour wall-clock limit for `main`; these launchers request that limit.
Training, validation, final TEST metrics, standalone inference, and plots share
the same 12-hour allocation. A complete 300-epoch run with these settings has
not been timed here. Epochs and evaluations are not shortened automatically,
and the launchers do not implement resume/requeue.

`legacy-scripts/` contains archived infrastructure for provenance/reference
only. Do not use those scripts for new formal experiments.

## Ad-hoc entrypoints and direct Python

Top-level `train_issm.sh` and `train_anuga.sh` are direct single-run entrypoints.
They default to `runtime/issm_fast.yaml` and `runtime/anuga_fast.yaml`, respectively,
with full-trajectory caching inherited from `configs/default.yaml`.
Set `RUNTIME_CONFIG=''` to omit the loader overlay; trajectory caching stays on.
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

For ISSM on Derecho, append the final override to keep a batch of four per rank
and accumulate two micro-batches per optimizer update:

```bash
bash train_issm.sh --config configs/runtime/issm_derecho.yaml
```

The command retains the ISSM loader overlay and uses four ranks by default,
giving an effective global batch of `4 ranks × 4 samples × 2 = 32`.
For another explicit ISSM config stack, append this YAML after all other
`--config` arguments so it takes precedence over earlier batch/accumulation
settings. ANUGA uses its own existing settings.

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

The CUDA smoke scripts also request a 12-hour Slurm walltime; their diagnostic
workload remains one epoch with four training batches per rank. Before the large
sweeps, run these jobs from an authenticated cluster login:

```bash
mkdir -p logs
sbatch tests/smoke_cuda_issm.sh
sbatch tests/smoke_cuda_anuga.sh
```

These are validation jobs, not formal results. They exercise real canonical
data/configuration, fixed B, four-rank training with sampled $k_{\mathrm{eff}}$, rollout
validation, best checkpoint, standalone evaluation, complete NPZ, and
postprocessing. Their bounded optimizer run does not constitute a formal
epoch or accuracy result. CUDA training is validated only when the actual jobs
complete and their evidence is inspected; CPU success alone does not establish it.
