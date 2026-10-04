# COGENT-NeuralODE handbook

This is the canonical guide for training and rollout evaluation. Detailed keys
and formulas are in [config_setting.md](config_setting.md); temporal loss details
are in [docs/temporal_consistency.md](docs/temporal_consistency.md).

## Configuration

```text
configs/
  default.yaml
  datasets/{issm,anuga,adcirc}.yaml
  protocols/{issm,anuga}/main.yaml
  models/node2.yaml
  ablations/
    history/h{1..8}.yaml
    future_len/k{30,45,60,64,75,90,120,150,180}.yaml
    rollout_start/known{8,60,90,120}.yaml
    architecture/encoder/{transformer,lstm}.yaml
    architecture/{residual,ode_context,relative_time}/{on,off}.yaml
    temporal_consistency/tc0_off.yaml ... tc5_hybrid.yaml
  runtime/fast.yaml
```

Merge order is **default → dataset → protocol → model → ablations → runtime**.
Later values win, dictionaries merge recursively, and lists replace wholesale.
Each ablation changes one scientific choice. Explicit default controls remain
available to make an experiment's config stack readable. Runtime settings
control loading performance and never select data or alter scientific settings.

| Quantity | Meaning |
| --- | --- |
| `dataset.history_len` (H) | Number of true context states immediately before the prediction start |
| `dataset.future_len` (K) | Maximum supervision available in each training window |
| `evaluation.known_steps` (S) | Absolute index of the first predicted state; prediction continues to trajectory end |

For trajectory length T, rollout history is `x[S-H:S]` and targets are `x[S:T]`.
Require `H <= S < T`. H6/known60 uses states 54–59; H6/known90 uses 84–89.
Training windows naturally vary with H and K. The sampled training horizon
`k_eff <= K` never limits inference. A K30 model still predicts 180 steps from
known60 on a 240-step ISSM trajectory.

| Protocol | Dataset | H | K | S | Relative-time scale | Batch | Min training horizon | Loss scale |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ISSM | `data/ISSM/PIG_5000` | 1 | 180 | 60 | 180 | 8 | 24 | 100 |
| ANUGA | `data/ANUGA/simulation_data_merged` | 1 | 64 | 8 | 65 | 1 | 8 | 1 |

Both use Transformer, residual decoding, history context in the ODE, relative
time, and TC off. The ISSM rate-modulo split and ANUGA random split remain in
the dataset YAMLs. The ANUGA path shown above is abbreviated; its dataset YAML
preserves the existing absolute data location. ADCIRC has a dataset adapter/config; select its temporal and
training settings explicitly because no ADCIRC scientific protocol is supplied.

## Training

Install `requirements.txt` in the intended PyTorch environment, then run:

```bash
bash train_issm.sh
bash train_anuga.sh
EXTRA_CONFIGS="configs/ablations/history/h6.yaml" bash train_issm.sh
```

`DATASET_CONFIG`, `PROTOCOL_CONFIG`, `MODEL_CONFIG`, `RUNTIME_CONFIG`, and
`EXTRA_CONFIGS` select explicit stack elements. `EXTRA_CONFIGS` is a
space-separated list. Launchers retain torchrun/process and environment controls;
inspect the script for local machine settings. To invoke Python directly:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/datasets/issm.yaml \
  --config configs/protocols/issm/main.yaml \
  --config configs/models/node2.yaml \
  --config configs/ablations/history/h1.yaml \
  --config configs/runtime/fast.yaml --run-name issm_h1
```

Every run writes `outputs/<run_name>/config.json`, `config_stack.txt`,
`split_files.json`, `history.json`, `best.pt`, and `final_metrics.json`.
The stack file lists config paths in their exact merge order. The checkpoint
contains the merged config, training normalizer, and exact split manifest.
Validation and final testing roll out from S to trajectory end; checkpoint
selection uses `whole_rollout_norm_rmse`. DDP shards scenarios without padding
and globally reduces error sums and element counts.

## Experiment order

1. **History first:** H1–H8 with K180, known60, all four default architecture
   choices, and TC0. Choose H* from rollout validation.
2. **Architecture:** hold H* and K180 fixed, run all 16 combinations of encoder,
   residual, ODE context, and relative time; keep known60 and TC0.
3. **Future supervision:** hold H* and selected architecture fixed; sweep
   K30/45/60/75/90/120/150/180. Every run still validates from known60 to the end.
4. **Temporal consistency:** hold selected H/K/architecture fixed, compare TC0–5.
5. **Rollout-start robustness:** evaluate the same final checkpoint from known60,
   known90, known120. Keep selected H and relative-time scale fixed.

Local sequential helpers and HPC submitters use the same explicit controls:

```bash
bash shell_scripts_sigspatial_issm/run_issm_history_scan_sequential.sh
bash sbatch_scripts_sigspatial_issm/submit_issm_history_scan.sh
bash sbatch_scripts_cercat_anuga/submit_anuga_history_scan.sh

# Set these from completed selection results before the later phases.
export HISTORY_LEN="$SELECTED_H"
bash shell_scripts_sigspatial_issm/run_issm_architecture_scan_sequential.sh
# HPC alternative: sbatch_scripts_sigspatial_issm/submit_issm_architecture_scan.sh

export ENCODER="$SELECTED_ENCODER" RESIDUAL="$SELECTED_RESIDUAL"
export ODE_CONTEXT="$SELECTED_ODE_CONTEXT" RELATIVE_TIME="$SELECTED_RELATIVE_TIME"
bash shell_scripts_sigspatial_issm/run_issm_future_len_ablation_sequential.sh
# HPC alternative: sbatch_scripts_sigspatial_issm/submit_issm_future_len_ablation.sh

export FUTURE_LEN="$SELECTED_K"
bash shell_scripts_sigspatial_issm/run_issm_temporal_consistency_scan_sequential.sh
# HPC alternative: sbatch_scripts_sigspatial_issm/submit_issm_temporal_consistency_scan.sh
```

Encoder accepts `transformer` or `lstm`; switches accept `on` or `off`.
Later-phase helpers require the selections; they do not assume a winning H.
TC is a separate phase, never a Cartesian product with architecture.

## Inference and saved results

`scripts/evaluate.py` is the sole formal evaluation entrypoint:

```bash
python scripts/evaluate.py --checkpoint outputs/final/best.pt --split test \
  --config configs/ablations/rollout_start/known60.yaml --output-dir outputs/rollouts
bash eval_rollout_start_sweep.sh --checkpoint outputs/final/best.pt \
  --output-dir outputs/rollouts
```

`eval_rollout_tmux.sh` wraps the same entrypoint. `--known-steps` can set the
start directly. Runtime overrides may relocate `dataset.data_dir` or select
AMP/device/output; checkpoint H, K, architecture, solver, and training settings
remain authoritative. There are no H/K inference overrides.

Each start requires a separate inference run with true history reanchored at S.
For T240, known60/90/120 predict 180/150/120 steps. Files use the stem
`best.<split>.known<S>.full_rollout`. The compressed predictions NPZ and metadata
JSON preserve every future node/channel, physical and normalized predictions and
targets, scenario/simulation IDs, history/start lengths, absolute indices,
adapter-provided time values, and lead steps. Plots may be shortened with
`--plot-max-lead-step`; complete predictions and whole-rollout metrics remain.
ANUGA exports can additionally create complete time series and flood maps.

## Post-processing without inference

```bash
python scripts/postprocess_rollout.py \
  --artifacts outputs/rollouts/best.test.known{60,90,120}.full_rollout_predictions.npz \
  --mode equal-lead --lead-steps 120 --output-prefix outputs/rollouts/equal120
python scripts/postprocess_rollout.py \
  --artifacts outputs/rollouts/best.test.known{60,90,120}.full_rollout_predictions.npz \
  --mode common-tail --output-prefix outputs/rollouts/common_tail
```

Equal lead compares the first 120 forecast steps: absolute intervals 60–179,
90–209, and 120–239. Common tail compares absolute 120–239 for all three runs,
revealing accumulated rollout-age effects on the same target period.
The postprocessor reads saved arrays, aligns scenarios/indices/times and mesh
structure, and writes JSON/CSV; it never loads a checkpoint or runs a model.
`--mode full`, `lead-slice`, and `absolute-slice` support further recomputation;
use `--help` for slice bounds. Keep the predictions NPZ and companion metadata
together as the authoritative inference output.

## Validation

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 python tests/check_temporal_ddp.py
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 python tests/check_rollout_ddp.py
python scripts/audit_foundation.py --help
python scripts/audit_relative_time.py --help
```

The [migration report](docs/config_evaluation_refactor_20261004.md) records the
cleanup and validation results. Older guidance and reports are retained only in
explicit historical directories and do not define the current workflow.
