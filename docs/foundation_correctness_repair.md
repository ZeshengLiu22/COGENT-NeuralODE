# COGENT-NeuralODE foundation correctness repair

Date: 2026-10-04. Starting repository state: `d4e655e` (cleaned NODE2 pipeline).

The requested correctness and protocol repairs are implemented. All **48 CPU tests pass**. Real-data audits confirm the ISSM forcing leak is removed, ANUGA rainfall normalizes to unit standard deviation, and relative-time rollout prefixes agree exactly on both full real meshes. No formal training was launched. The only optimization runs used tiny synthetic trajectories.

## Behavior before and after

| Area | Before | After |
|---|---|---|
| ISSM information | Static coordinates/base/surface; forcing included evolving floating values. | Static `[base0, surface0, speed0, initial_mask]`; state `[vx, vy, thickness]`; forcing `[initial-mask basal melt, SMB(t)]`. Binary mask 1 means floating/ocean (`floating[0] < 0`). No future floating reaches inputs. Coordinates stay local to auxiliary edge construction. |
| ISSM split | Rate-modulo split. | Unchanged: validation rates 0/20/40/60, test 10/30/50/70, remaining rates training. |
| Direct paper comparison | H1 sliding windows could still use later observed states. | `configs/issm_paper_matched.yaml`: H1, K239, a single t_end=0 origin on the 240-snapshot trajectories, known steps 1, fixed scale239. Standard history scans are explicitly a different history-conditioned forecasting protocol. |
| Normalization | Std below 1e-5 became denominator1, suppressing real rain variability. | Centered float64 running mean/variance. Only std <= float64 epsilon × abs(mean) is numerically constant; those channels use denominator1. Otherwise the true population std is used. Final serialized stats/model tensors remain FP32. `std_floor` is retained only as inert metadata. Rain stays in m/s. |
| Relative time | `t / eval_times[-1]`, with endpoint dependence. | `t / relative_time_scale`, fixed for the trained model, finite and positive when enabled. Standard ISSM180, ANUGA65; model overlay does not overwrite dataset scale. No horizon-dependent clipping. |
| ODE backpropagation | Standard/adjoint branch and config switch. | Ordinary `odeint` only. Active model/configs/examples have no functional adjoint setting. |
| Checkpoint provenance | Standalone window evaluation could adopt external config and rediscover splits. | `best.pt` embeds config, normalizer, and relative `split_manifest`; separate `split_files.json` retained. Evaluation uses saved model/solver settings, rejects unauthorized config changes, and resolves the exact saved membership after relocation. Missing/duplicate scenarios fail clearly. New files cannot enter the split. |
| Ablation origins | H/K changed window counts and prediction origins. | Optional `dataset.window_reference` uses max current/reference H and K. All 24 formal variants have common anchors within their scans. Absent reference preserves old enumeration exactly. |
| Model selection | Aggregate physical rollout RMSE mixed units. | Active formal/default selection uses `whole_rollout_norm_rmse`. Physical metrics remain fully reported. |
| ISSM reporting | Component errors only. | Adds physical derived speed RMSE in m/yr and thickness RMSE in m; no new loss or speed-based selection. |
| Distributed evaluation | Standard sampler could pad duplicate samples. | Evaluation indices are `range(rank, N, world_size)` without dropping/padding. Evaluator bypasses DDP forward broadcasts for unequal shard lengths, then globally reduces SSE/count. Full-rollout scenario sharding retained. |
| Accumulation | Tail loss divided by configured group size. | Divides by actual number of batches in each group; one synchronized horizon remains shared per group. Three-batch/two-step regression produces gradients -4 and -10 as expected. |
| Loader workers | Persistent workers could retain stale resampled windows. | Training persistence automatically disabled with a warning only when workers>0 and epoch window resampling is configured. Formal overlays keep persistent workers. |
| Evaluation precision | Trainer reused training BF16; standalone used FP32. | Independent `evaluation.amp_mode: none` default. Training remains BF16; selection/final/standalone evaluation default to FP32. |
| Midpoint validation | No refinement diagnostic. | New standalone validation-only script compares saved/default grid, step0.5, step0.25, writes error/difference/runtime JSON, and restores solver settings. |

`edge_attr` is not consumed by NODE2; edge handling was not redesigned. The four static + three state + two forcing + one time inputs give the requested ten semantic inputs for the initial-state protocol. Longer histories intentionally add observations. Old ISSM checkpoints and old three-force prepacked ISSM data are incompatible with the new protocol and require retraining/regeneration; no compatibility path was added.

Transformer/LSTM architecture and the sole `history_encoder_type` selector are preserved, including position encoding, layers, heads, feed-forward size, dropout, pooling, and bidirectional attention over observed history. A parsed-YAML comparison found no unrelated hyperparameter changes in the 27 modified existing configs. No temporal-consistency loss, new NODE variant, removed-model restoration, scientific-paper edit, or main-loss change was introduced.

## Automated tests and reproducibility

Environment: CPU, Python3.11, PyTorch2.8.0+cu128; CUDA was unavailable. Command from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/work/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python \
-m unittest discover -s tests -v
```

Final result:

```text
Ran 48 tests in 8.184s

OK
```

The [complete unabridged suite output](repair_validation/cpu_tests.txt) contains all tests and evaluator logs. Coverage: 16 existing smoke tests, 14 dataset regressions, 7 model/config regressions, 7 checkpoint/CLI/solver regressions, and 4 training/precision/metric/DDP-forward regressions. `git diff --check` and Python compilation of all active source/test/experiment modules also passed.

Targeted checks cover all requested leakage, mask, dimensions, state ordering, rate split, tiny/constant/ordinary variance, normalizer round-trip, relative-time ON prefix equality (linear and backward-Hermite forcing), immutable model/solver configuration, data relocation/new-file split stability, pairwise split disjointness, common anchors, sampler coverage/disjointness, incomplete accumulation, FP32 defaults, and standard-only ODE backpropagation. Smoke tests now use distinct scenario identifiers for train/validation/test.

Active-code audit command:

```bash
rg -n 'use_adjoint|odeint_adjoint|use_transformer_history' models configs training scripts experiments
```

Result: no matches. Historical design records remain historical; they do not provide functional settings. `configs/model_node2.yaml` was audited and already leaves the dataset-specific scale intact, so no edit was needed there.

## Real-data audit

Complete evidence, per-file lengths, rates, leakage errors, and every scan row: [foundation_audit.json](foundation_audit.json). Reproduce without training:

```bash
python scripts/audit_foundation.py \
  --data-root /scratch/09575/zeshengliu/COGENT-NeuralODE/data \
  --output docs/foundation_audit.json
```

The formal launch scripts apply the PIG-5000 fast-loader overlay after their base/dataset/model configs. The audit distinguishes that dataset from the broader default `PIG_data` directory.

| Dataset | Scenarios (train/val/test) | Snapshots | Nodes | Full future / fixed scale |
|---|---:|---:|---:|---:|
| Formal ISSM PIG_5000 | 28/4/4 | 240, all 36 files verified | 2,852 | 180 / 180 |
| Generic ISSM PIG_data | 84/12/12 | 240, all 108 files verified | Multiple meshes; sampled 6,384 | 180 / 180 |
| Formal ANUGA simulation_data_merged | 12/4/4 | 73, all 20 files verified | 68,464 | 65 / 65 |

Formal ISSM trajectory shapes: static `[2852,4]`, forcing `[240,2852,2]`, state `[240,2852,3]`. The generic ISSM sample shapes are `[6384,4]`, `[240,6384,2]`, `[240,6384,3]`. Static columns are initial base/surface/speed/binary floating-ocean mask; forcing columns are masked basal melt and SMB; state columns remain vx/vy/thickness. Future floating is absent.

Published rate sets remain exact. Training rates are 2,4,6,8,12,14,16,18,22,24,26,28,32,34,36,38,42,44,46,48,52,54,56,58,62,64,66,68; validation 0,20,40,60; test 10,30,50,70. The generic directory repeats these rates across three meshes.

The old raw leakage identity was rerun in float32 on every validation/test scenario at t_end59, across the remaining 180 snapshots. `H[59] + floating[60:] - floating[59]` still reconstructs raw future thickness with RMSE **7.68108e-5..7.84962e-5** on formal PIG-5000 and **7.23657e-5..8.44815e-5** across generic PIG_data. This source-data relationship remains; its future floating operand is no longer accessible to the model. Regression tests perturb future floating and compare every produced sample tensor.

ANUGA rainfall statistics use all training times and nodes, with exactly equivalent node weights. The audit reads rainfall/time/mesh metadata only, fits the actual FeatureNormalizer with compact repeated rainfall, and avoids loading all water-state arrays. No rainfall unit conversion occurs.

| Rainfall statistic | Value |
|---|---:|
| Raw mean [m/s] | 5.862700872449432e-6 |
| Raw population std [m/s] | 5.8554946292813445e-6 |
| Raw minimum [m/s] | 0 |
| Raw maximum [m/s] | 1.9991824956377968e-5 |
| Normalized mean | -4.105101931821398e-8 |
| Normalized population std | 0.9999999812840193 |

## Relative-time prefix audit

[Real-prefix evidence and exact model settings](repair_validation/real_prefix.json) were generated by `python scripts/audit_relative_time.py --data-root /scratch/09575/zeshengliu/COGENT-NeuralODE/data --output docs/repair_validation/real_prefix.json`.

| Sample | Mesh nodes | Fixed scale | K4 vs K8-prefix max absolute difference | Prefix RMSE difference |
|---|---:|---:|---:|---:|
| ISSM PIG_data, melt rate2 | 6,384 | 180 | 0 | 0 |
| ANUGA sim0015 | 68,464 | 65 | 0 | 0 |

Both normalized and physical predictions agree exactly. These are full-graph CPU inference checks with seed 31, H8, relative time ON, backward-Hermite forcing, midpoint, an untrained compact 16 model, and a normalizer fitted on the selected training trajectory solely for the audit. Production architectures/configured sizes were not changed or tuned. These checks establish prefix consistency, not forecast accuracy.

## Common-window verification tables

Indices below are zero-based history endpoints (`t_end`). Each formal table uses the launch-script fast-loader overlay. ISSM has 28 training scenarios and per-rank batch 8; ANUGA has 12 training scenarios and batch 1. All variants retain accumulation 1. Steps are optimizer updates per rank per epoch, shown for one process and for the formal default four ranks. Training sampler padding remains unchanged.

The general count is `ceil(ceil(ceil(total_windows / W) / batch_size) / grad_accum_steps)`. These are deterministic enumeration/loader counts, not measured formal training runs. Full anchor-list equality is asserted for every scenario and variant.

### ANUGA history scan: reference H8/K64

| Config filename | H | K | Windows/scenario | First t_end | Last t_end | Steps W=1 | Steps W=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_ANUGA_history1.yaml` | 1 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history2.yaml` | 2 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history3.yaml` | 3 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history4.yaml` | 4 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history5.yaml` | 5 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history6.yaml` | 6 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history7.yaml` | 7 | 64 | 2 | 7 | 8 | 24 | 6 |
| `base_ANUGA_history8.yaml` | 8 | 64 | 2 | 7 | 8 | 24 | 6 |

### ISSM history scan: reference H8/K120

| Config filename | H | K | Windows/scenario | First t_end | Last t_end | Steps W=1 | Steps W=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_ISSM_history_1.yaml` | 1 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_2.yaml` | 2 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_3.yaml` | 3 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_4.yaml` | 4 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_5.yaml` | 5 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_6.yaml` | 6 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_7.yaml` | 7 | 120 | 113 | 7 | 119 | 396 | 99 |
| `base_ISSM_history_8.yaml` | 8 | 120 | 113 | 7 | 119 | 396 | 99 |

### ISSM future-length scan: reference H6/K180

| Config filename | H | K | Windows/scenario | First t_end | Last t_end | Steps W=1 | Steps W=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_ISSM_history6_future30.yaml` | 6 | 30 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future45.yaml` | 6 | 45 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future60.yaml` | 6 | 60 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future75.yaml` | 6 | 75 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future90.yaml` | 6 | 90 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future120.yaml` | 6 | 120 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future150.yaml` | 6 | 150 | 55 | 5 | 59 | 193 | 49 |
| `base_ISSM_history6_future180.yaml` | 6 | 180 | 55 | 5 | 59 | 193 | 49 |

Targets still contain the requested K steps. The common prediction origins make training-window and update budgets equal within each scan. Removing a reference constraint restores the original `T-H-K+1` window rule.

## Synthetic training smoke and standalone reproduction

[Complete smoke evidence](repair_validation/synthetic_training_smoke.json) records the commands, metrics, training curves, and solver diagnostics. Each dataset used three nodes, eight snapshots, one distinct scenario per split, Transformer H2/K2, CPU FP32, and five tiny epochs. Both runs were finite, selected epoch 5, saved/reloaded `best.pt`, executed full rollout, and successfully invoked the actual standalone window, full-rollout, and convergence scripts.

| Synthetic dataset | Training normalized MSE, epoch1 → epoch 5 | Largest recorded Trainer/standalone metric difference |
|---|---:|---:|
| ANUGA | 1.11127998 → 0.80558349 | 5.61294e-8 |
| ISSM | 0.88155845 → 0.54839897 | 1.55866e-8 |

Roundoff reflects SSE accumulation order; all model inference used FP32. Smoke artifacts, configs, checkpoints, and `solver_convergence.json` are retained under ignored `outputs/foundation_smoke/{anuga,issm}`. The original execution paths in the evidence refer to the writable repair checkout.

## Metrics, evaluation, and solver use after retraining

Formal selection uses normalized overall rollout RMSE. Reports continue to contain physical RMSE/MAE, per-channel physical RMSE/MAE, and physical horizon RMSE curves, plus their normalized counterparts. ISSM adds `speed_rmse_m_per_yr` and `thickness_rmse_m`; full/final summaries use `whole_rollout_`/`final_step_` prefixes. Speed is derived from inverse-normalized vx/vy and is reporting-only.

Standalone evaluation needs only the checkpoint; use explicit runtime flags as required:

```bash
python scripts/evaluate.py --checkpoint outputs/RUN/best.pt --device cpu
python scripts/run_full_rollout.py --checkpoint outputs/RUN/best.pt --device cpu
python scripts/check_solver_convergence.py --checkpoint outputs/RUN/best.pt --device cpu
```

`--data-dir NEW_ROOT` relocates saved relative file identifiers without rediscovering splits. `evaluate.py` permits explicit history/future length, batch-size, worker, device, AMP, and output-location overrides. External model/encoder/time/solver/protocol changes raise an error. Saved split membership must remain disjoint. Full rollout uses the checkpoint manifest by default; an explicit legacy JSON manifest is validated against saved membership.

For the direct initial-information comparison, use `configs/issm_paper_matched.yaml` followed by `configs/model_node2.yaml`, optionally the data/loader overlay. Omit the standard `configs/issm.yaml` overlay, which sets the separate 60-known-step forecasting protocol. The direct protocol uses K239/scale239 to preserve a single initial prediction origin; standard formal forecasting stays at the verified scale180.

The solver diagnostic uses saved config/normalizer/validation membership and FP32 full rollouts. It reports physical and normalized validation RMSE, coarser-to-next-finer prediction RMSE/max differences, and runtime for the saved/default grid, midpoint0.5, and midpoint0.25. Half-to-quarter normalized changes were 0.194% of model error for synthetic ANUGA and 0.530% for synthetic ISSM. Those values validate the diagnostic only. Numerical adequacy for a newly trained real model must be assessed on its validation split after clean retraining. No formal solver setting was changed based on test accuracy.

## Remaining limitations

No implementation blocker remains from this repair pass. CUDA was unavailable, so GPU BF16/FP32 numerical comparisons were not executed. An actual multi-process DDP run was also not executed; CPU tests verify precision selection, rank-index partitioning, and bypass of DDP forward collectives. No formal accuracy or real-trained-model convergence claim is made. Old ISSM checkpoints need clean retraining. Historical design documents remain historical and may describe removed experiments; the active runtime/configs and this report define the repaired protocol.

## Exact changed source/config/test/report files

65 files relative to the cleaned commit (ignored smoke artifacts are listed above):

```text
config_setting.md
configs/ANUGA_History_Scan/base_ANUGA_history1.yaml
configs/ANUGA_History_Scan/base_ANUGA_history2.yaml
configs/ANUGA_History_Scan/base_ANUGA_history3.yaml
configs/ANUGA_History_Scan/base_ANUGA_history4.yaml
configs/ANUGA_History_Scan/base_ANUGA_history5.yaml
configs/ANUGA_History_Scan/base_ANUGA_history6.yaml
configs/ANUGA_History_Scan/base_ANUGA_history7.yaml
configs/ANUGA_History_Scan/base_ANUGA_history8.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future120.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future150.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future180.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future30.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future45.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future60.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future75.yaml
configs/ISSM_Future_Len_Ablation/base_ISSM_history6_future90.yaml
configs/ISSM_History_Scan/base_ISSM_history_1.yaml
configs/ISSM_History_Scan/base_ISSM_history_2.yaml
configs/ISSM_History_Scan/base_ISSM_history_3.yaml
configs/ISSM_History_Scan/base_ISSM_history_4.yaml
configs/ISSM_History_Scan/base_ISSM_history_5.yaml
configs/ISSM_History_Scan/base_ISSM_history_6.yaml
configs/ISSM_History_Scan/base_ISSM_history_7.yaml
configs/ISSM_History_Scan/base_ISSM_history_8.yaml
configs/anuga.yaml
configs/base_sample.yaml
configs/issm.yaml
configs/issm_paper_matched.yaml
datasets/anuga_dataset.py
datasets/base_dataset.py
datasets/factory.py
datasets/issm_dataset.py
datasets/normalization.py
datasets/split_utils.py
datasets/window_utils.py
docs/continuous_graph_emulator_master_handbook.md
docs/foundation_audit.json
docs/foundation_correctness_repair.md
docs/repair_validation/cpu_tests.txt
docs/repair_validation/real_prefix.json
docs/repair_validation/synthetic_training_smoke.json
experiments/dataloader_speed_sweep/demo_defaults.yaml
experiments/dataloader_speed_sweep/formal_train_smoke_fullhorizon.yaml
experiments/dataloader_speed_sweep/formal_train_smoke_h60.yaml
experiments/dataloader_speed_sweep/run_sweep.py
experiments/dataloader_speed_sweep/train_timed.py
models/continuous/node_latent_block.py
models/node2_model.py
scripts/audit_foundation.py
scripts/audit_relative_time.py
scripts/check_solver_convergence.py
scripts/evaluate.py
scripts/run_full_rollout.py
scripts/train.py
tests/test_foundation_checkpoint.py
tests/test_foundation_datasets.py
tests/test_foundation_model.py
tests/test_foundation_training.py
tests/test_smoke.py
training/evaluator.py
training/metrics.py
training/trainer.py
utils/checkpoint_evaluation.py
utils/eval_artifacts.py
```
