# Continuous graph emulator: master technical handbook

This is the current end-to-end reference for the active NODE2 implementation.
It recovers useful data, model, and training explanations from the earlier
design handbook and reconciles them with repaired adapters, fixed time scales,
temporal objectives, rollout-only evaluation, and the formal series budget.
The [operating handbook](../handbook.md) gives launch commands; the
[configuration reference](../config_setting.md) describes configuration ownership.

## 1. Scientific problem and active pipeline

A simulation is a trajectory on a fixed graph
\(\mathcal G=(\mathcal V,\mathcal E)\). Node \(i\) has static features
\(s_i^{raw}\), observed state \(y_i(q)\), and time-varying external forcing
\(u_i(q)\). The task is to predict a complete future state series from a short
observed history and known future forcing. Each model call initializes one
latent trajectory and integrates it forward; decoded predictions are not fed
back through the history encoder at every future snapshot.

```text
raw simulation files
  -> scenario split and saved split identity
  -> normalization fitted on training trajectories only
  -> natural H/K-specific training-anchor enumeration
  -> fixed per-scenario epoch series sampling
  -> H-step normalized state/forcing history
  -> static node encoder
  -> shared graph encoder at each history step
  -> per-node Transformer or LSTM history encoder
  -> latent initialization
  -> continuous interpolation of known forcing
  -> latent graph Neural ODE
  -> direct or residual state decoder
  -> complete-series state loss and optional temporal consistency
  -> rollout-to-end validation, best checkpoint, rollout-to-end test
  -> complete saved rollout arrays and model-free postprocessing
```

NODE2 is the supported runtime model. Earlier NODE1/NCDE designs and deferred
extensions are historical material, not selectable active model paths. Using
`torchcde` to interpolate forcing does not turn NODE2 into an NCDE: the vector
field uses forcing values \(u(t)\), not a learned derivative against \(du(t)\).

## 2. Temporal notation and available information

All snapshot indices are zero-based; Python slice stops are exclusive.

| Symbol | Meaning | Configuration or source |
| --- | --- | --- |
| \(T\) | Number of stored snapshots in one simulation | Trajectory |
| \(H\) | Number of true historical states supplied to the encoder | `dataset.history_len` |
| \(K\) | Maximum future block stored for a training series | `dataset.future_len` |
| \(k_{eff}\) | Actual sampled prediction length in one optimization step | Horizon sampler, at most K |
| \(S\) | Known-prefix length and absolute index of the first predicted state | `evaluation.known_steps` |
| \(B\) | Number of selected training series per scenario per epoch | `dataset.train_series_per_scenario_per_epoch` |

At a training history anchor \(a\), inputs and targets are

\[
Y_{hist}=Y[a-H+1:a+1],\qquad
U_{hist}=U[a-H+1:a+1],
\]

\[
U_{future}=U[a+1:a+K+1],\qquad
Y_{future}=Y[a+1:a+K+1].
\]

The model consumes known future forcing; future state is a loss target and is
never an input. After horizon sampling, training predicts every state
\(\hat y(a+1),\ldots,\hat y(a+k_{eff})\). It is not endpoint-only regression.

Formal validation, final test, and standalone inference use

\[
Y_{hist}=Y[S-H:S],\qquad
Y_{target}=Y[S:T],\qquad H\le S<T.
\]

The last observed anchor is \(S-1\), and prediction length is \(T-S\).
The known prefix can be longer than H; the encoder sees its final H states.
For ISSM T240/H4/known60, history is 56–59 and prediction is 60–239.
A model trained with K30 still predicts those 180 future states. H and K remain
checkpoint-authoritative at inference; only S is an allowed temporal override.

## 3. Scenario adapters, graph construction, and splits

`TrajectoryData` in `datasets/base_dataset.py` is the shared adapter contract:

| Array | Shape | Interpretation |
| --- | --- | --- |
| `x_static` | `[N, F_static]` | Node features constant throughout this trajectory |
| `force` | `[T, N, F_force]` | Known external forcing |
| `state` | `[T, N, F_state]` | Supervised physical state |
| `edge_index` | `[2, E]` | Node connectivity |
| `times` | `[T]` | Strictly increasing adapter-provided time coordinates |
| `scenario_id`, `sim_id` | Strings | Scenario and simulation identity |

Validation checks dimensions, finite values, matching node/time counts, and
strict time ordering. A spatially uniform `[T, F_force]` forcing array can be
broadcast over nodes. Optional edge attributes, scenario parameters, and target
masks are carried as metadata. Current NODE2 graph layers consume `edge_index`;
optional edge attributes and scenario parameters are not additional inputs
to its vector field. The active state/TC objectives do not apply target masks.

### ISSM

The formal dataset is `data/ISSM/PIG_5000`. The cell-format adapter supplies:

| Group | Channels in order |
| --- | --- |
| Static | Initial base, initial surface, initial speed, initial floating/ocean mask |
| Forcing | Basal melt multiplied by the initial mask, SMB at the queried time |
| State | vx, vy, thickness |

The initial mask equals one where `floating[0] < 0`. The adapter does not expose
future floating values. Coordinates construct auxiliary static edge data,
not the four static model channels. MATLAB triangle indices are converted to
zero-based graph indices; triangular connectivity produces deduplicated edges
in both directions. Initial speed is \(\sqrt{v_x(0)^2+v_y(0)^2}\).

The formal rate split uses remainder zero modulo 20 for validation and remainder
ten for test; remaining rates train. The current formal files give 28/4/4
train/validation/test scenarios, each with T240. Generic ISSM containers must
honor the same four-static/two-forcing/three-state channel contract.

### ANUGA

The configured dataset path ends in `ANUGA/simulation_data_merged`. Static
channels are x, y, elevation, and friction. Forcing is rainfall in its supplied
SI m/s units, broadcast over nodes. State channels are water depth, x momentum,
and y momentum. Supplied `h` is used when available; otherwise depth is
\(\max(stage-elevation,0)\). Stored connectivity is used directly or constructed
from triangular volumes.

The dataset YAML specifies a seeded random split with fractions 0.6/0.2/0.2
and seed 42. The current 20 simulations give 12/4/4 scenarios, each with T73.
ANUGA preserves its supplied simulation time array. The ISSM cell-format adapter
uses snapshot-index times; exports do not invent physical units for those indices.

### Split identity and ADCIRC scope

Splitting occurs before training-series expansion. A saved manifest records
exact train/validation/test membership; evaluation restores it instead of
discovering and randomly splitting files again. Relocating the data root
preserves saved relative file identities and requires disjoint splits.

ADCIRC retains a dataset adapter and dataset configuration. There is no formal
ADCIRC main protocol or launcher sweep in this cleanup; temporal/training
choices must be supplied explicitly for an ad-hoc run.

## 4. Normalization and solver time

`FeatureNormalizer` estimates separate per-channel means and population
standard deviations for static, forcing, and state arrays using all training
trajectories. It does not fit on selected epoch anchors or on validation/test.
Dynamic statistics aggregate time and node elements; static statistics
aggregate nodes. Centered float64 running statistics retain small real
variances such as ANUGA rainfall.

\[
\tilde x_c=(x_c-\mu_c)/\sigma_c,\qquad
x_c=\tilde x_c\sigma_c+\mu_c.
\]

Numerically constant channels use denominator one. Detection uses
\(\sigma_c\le\epsilon_{64}|\mu_c|\), not a cutoff in physical units.
`normalization.std_floor` remains serialized metadata and does not replace
small nonzero standard deviations. Serialized statistics and model tensors
are float32. Every evaluation restores the checkpoint normalizer.

Solver coordinates are distinct from stored absolute indices and times. If
\(q_j\) denotes adapter time and \(a\) the current anchor, the dataset constructs

\[
\tau_j=\frac{q_j-q_a}{\operatorname{median}(q_{m+1}-q_m)}.
\]

The denominator uses positive spacings over the trajectory, so a uniform grid
becomes history offsets ending at 0 and future offsets 1,2,… . It does not depend
on the requested prediction endpoint. Adapter times remain in `future_time`.
Batched trajectories must share their solver time grid; use batch size 1 or
align grids if their relative spacings differ.

## 5. Natural anchors and fixed epoch exposure

At stride 1, legal history anchors are

\[
\mathcal A(H,K,T)=\{H-1,H,\ldots,T-K-1\},\qquad
N_{\mathrm{natural}}=T-H-K+1
\]

when \(T\ge H+K\). A shorter trajectory has no legal anchor. The formal
protocols use stride 1. The generic stride option takes every `stride`th anchor
from this natural range. Each H/K variant uses its own legal range; anchors
are not restricted to an intersection with another variant.

One selected training series is identified by a simulation/scenario and a
natural history anchor. Its actual prediction length is determined later by
the independently sampled `k_eff`. The internal `WindowMetadata` type is only
a representation of that identity; it does not define formal evaluation.

For each scenario independently, let N be its legal-anchor count and B the
configured epoch budget:

1. **N ≥ B:** select exactly B distinct anchors without replacement. N=B uses
   every anchor once; N>B permits different subsets in different epochs.
2. **0 < N < B:** include every legal anchor once, draw B−N additional anchors
   with replacement from that same legal set, and shuffle the scenario's list.
3. **N=0:** a positive budget cannot be fulfilled; report the invalid training
   setup instead of silently removing the scenario's contribution.

Epoch selection is reproducible from dataset seed and epoch. All DDP ranks
construct the same active dataset before the distributed sampler shards its
indices. Epoch changes can alter subsets, repetitions, or ordering. No shared
cross-variant subset is imposed; when N=B the selected set is naturally complete.

The budgets preserve the natural H1/full-horizon exposure:

| Dataset | Canonical natural count | B per scenario | Train scenarios | Series per epoch |
| --- | --- | ---: | ---: | ---: |
| ISSM | 240−1−180+1 = 60 | 60 | 28 | 1,680 |
| ANUGA | 73−1−64+1 = 9 | 9 | 12 | 108 |

These are protocol choices derived from canonical exposure, not extra tuned
hyperparameters. Every formal H/K variant retains the same dataset length.
Both totals are divisible by four, so the four-rank training sampler requires
no padding: each ISSM rank receives 420 series and each ANUGA rank receives 27.
With batch sizes 8 and 1, accumulation 1, and the retained final partial batch,
this gives 53 and 27 optimizer steps per rank per epoch, respectively. These
are loader counts, not measured runtimes.

Concrete ANUGA T73 examples at K64:

| H | Natural anchors | N | B=9 selection |
| ---: | --- | ---: | --- |
| 1 | 0,…,8 | 9 | Every anchor exactly once |
| 4 | 3,4,5,6,7,8 | 6 | All six once plus three replacement draws |
| 8 | 7,8 | 2 | Both once plus seven replacement draws |

For H4, the multiset `3,4,5,6,7,8,4,7,8` is valid before final shuffling.
For H8, `7,8,7,8,8,7,7,8,7` is valid. Repeated anchor 7 can participate in
separate optimization steps with k_eff 12, 40, 55, supervising states 8–19,
8–47, and 8–62 respectively. A repeated anchor does not fix supervised series
length, although equal k_eff draws remain possible.

Fixing B controls scenario exposure and epoch sample count. It does not
equalize unique anchors, total target timesteps, k_eff, series ending time,
rollout computational cost, or history length. Normalization and elementwise
loss reductions retain their existing weighting; equal series counts do not
turn them into equal-per-scenario loss reductions for different-sized graphs.

Only training constructs natural-anchor lists. Validation, test, and standalone
evaluation construction keep no training-anchor list and do not expand
training windows. They still expose `scenario_infos`, `iter_trajectories()`,
`get_rollout_data()`, normalization, scenario identity, and mesh metadata.
Their series-indexed dataset length is not their scenario count; trajectory
evaluation uses `scenario_infos` directly.

## 6. Tensor construction and PyG batching

Selected samples are transformed from time-major raw arrays to node-major
model tensors. Let \(N_\Sigma\) be total nodes and Q the graph count in a batch:

| Tensor | Before horizon truncation | At a training forward |
| --- | --- | --- |
| `x_static` | `[N_sum, F_static]` | Unchanged |
| `state_hist` | `[N_sum, H, F_state]` | Unchanged |
| `force_hist` | `[N_sum, H, F_force]` | Unchanged |
| `force_future` | `[N_sum, K, F_force]` | `[N_sum, k_eff, F_force]` |
| `y_future`, prediction | `[N_sum, K, F_state]` target | `[N_sum, k_eff, F_state]` |
| `t_future`, `future_idx`, `future_time` | `[Q, K]` | `[Q, k_eff]` |
| `batch` | `[N_sum]` | Node-to-graph assignment |

PyG batches disconnected graphs and offsets connectivity, so message passing
does not mix simulations. History encoding preserves node identity and
processes only the H positions for each node.

The generic fast runtime selects workers, pinned memory, and prefetching;
`cache_in_memory` remains false. Training resampling requires worker copies to
receive the updated series list each epoch, so the loader disables persistent
workers when necessary even if requested in the runtime YAML. This prevents
stale worker-side anchor selections.

## 7. Static, graph, and temporal encoders

Let \(s_i=E_s(\tilde s_i^{raw})\) be the static MLP embedding. At each observed
history step \(\ell\), a shared spatial graph network computes

\[
h_i^\ell=G_\phi([\tilde y_i^\ell,\tilde u_i^\ell,s_i],\mathcal G).
\]

The same graph encoder weights are reused at every history step. Its outputs
stack as `[N_sum,H,D_enc]`. The canonical temporal encoder is a Transformer
with sinusoidal positions and last-token pooling:

\[
c_i=P\left(\operatorname{Transformer}
(h_i^1+p_1,\ldots,h_i^H+p_H)_H\right).
\]

Projection P is identity when encoder and context widths match. Mean pooling
is supported; formal full architecture retains last pooling. The LSTM choice
uses its final top-layer hidden state as c_i. Attention operates across
observed time for each node, without a causal mask because all H inputs are
known. Node chunking retains each node's entire history. CUDA math attention
is the default for the temporal Transformer. Initialization always uses context:

\[
z_i(0)=\operatorname{InitMLP}([c_i,\tilde y_i^{last},s_i]).
\]

The model YAML uses latent/static/context width 96, two graph layers, and
Transformer two layers/four heads/feed-forward width 384. Factorial overlays
explicitly set the four tested choices and leave these dimensions unchanged.
See [architecture equations](upgrade_v1.md).

## 8. Forcing interpolation and latent graph ODE

The forcing path starts at 0 with the last observed forcing and continues
through all retained future forcing values:

\[
\mathcal U=\{(0,\tilde u^{last}),(\tau_1,\tilde u_1),\ldots,
(\tau_L,\tilde u_L)\},
\]

where L=k_eff in training and L=T−S in evaluation. The interpolator provides
\(u(t)\) at solver query times. The model uses `hermite_cubic_backward`;
linear interpolation is also implemented. Future states never construct this path.

With both optional ODE inputs enabled, NODE2 integrates

\[
\frac{dz_i}{dt}=f_\theta(z_i(t),u_i(t),s_i,c_i,r(t),\mathcal G),
\qquad r(t)=t/\tau_{\mathrm{scale}}.
\]

The protocol sets \(\tau_{\mathrm{scale}}=180\) for ISSM and 65 for ANUGA.
These fixed model constants are independent of K, S, and requested endpoint.
The time feature is not clamped. Turning ODE context off removes c_i from the
vector field but preserves it in initialization. Turning relative time off
removes only r(t).

`odeint` receives `[0,*t_future]`; the returned initial latent is dropped,
leaving every requested future latent. The model uses standard ODE
backpropagation. Initial-latent computation, interpolation, and integration
execute in float32 with autocast disabled. The surrounding encoder/decoder can
use training AMP; formal evaluation defaults to float32. Midpoint, its saved
options, interpolation, and tolerances belong to model configuration. Solver
convergence should be assessed on validation data for a trained model; prefix
equality alone establishes no prediction accuracy.

## 9. Decoder, state objective, and optional TC

For direct decoding, \(\hat y_i(t)=D_\psi(z_i(t))\). Canonical residual decoding
uses normalized state space:

\[
\Delta\hat y_i(t)=D_\psi(z_i(t)),\qquad
\hat y_i(t)=\tilde y_i^{last}+\Delta\hat y_i(t).
\]

The same last-observed anchor is broadcast over every future position; it is
not a sum of predicted adjacent increments. Decoder chunking splits flattened
node/time rows to reduce memory. Outputs are inverse-normalized for metrics.

Let a be `training.loss_scale_factor`, and define loss-space tensors
\(\hat Y=a\hat y\), \(Y=a\tilde y\). The current objective is

\[
L_{\mathrm{state}}=\frac{1}{N_\Sigma k_{eff}F}
\sum_{i,j,c}(\hat Y_{ijc}-Y_{ijc})^2,\qquad
L_{\mathrm{total}}=L_{\mathrm{state}}+\lambda_{\mathrm{TC}}L_{\mathrm{TC}}.
\]

ISSM uses a=100 and ANUGA a=1. Scaling tensors by a multiplies MSE by \(a^2\).
TC uses the same scaled tensors and solver time grid. The six supplied choices
are off, adjacent increment, random-pair increment, multiscale rate, rate plus
curvature, and hybrid. They reuse the same prediction; there is no second
forward or automatic TC magnitude balancing. Equations, reduction dimensions,
RNG rules, and short-horizon behavior are in [temporal consistency](temporal_consistency.md).

## 10. Epoch budget, curriculum, and optimization

The execution order separates two independent selections:

1. Select B scenario/anchor series per simulation for the epoch.
2. Construct an H-history/K-future sample for each selected series.
3. Resolve the epoch curriculum cap from K and training horizon settings.
4. Sample k_eff at the start of an optimizer/accumulation group and broadcast
   it from DDP rank 0 to every rank.
5. Truncate all future-aligned tensors and metadata to k_eff.
6. Predict every future state and compute state plus optional TC loss.
7. Backpropagate and update AdamW; run rollout validation when scheduled.

The default horizon distribution is biased toward long horizons. Null
`train_horizon_max` resolves to K. ISSM's minimum is 24; ANUGA's minimum is 8,
so the ANUGA K8 ablation is valid. The curriculum is a staircase over 120 epochs
with fractions 0.40/0.55/0.70/0.85, then the full target cap. Exact half-up
rounding, stage indexing, and probabilities are in [horizon curriculum](upgrade_v1.1.md).

With accumulation greater than one, microbatches in the same optimizer group
share k_eff; the final partial group divides by its actual size. Default
accumulation is 1. The scheduler steps once per epoch; fixed series exposure
makes the epoch's sample/update budget comparable within each formal sweep.
This does not equalize ODE work across k_eff or history lengths.

`train_norm_mse` measures unscaled normalized prediction error.
`train_loss` aliases `train_total_objective` before accumulation division.
State, raw TC, weighted TC, total, and TC components are logged separately.
Prediction metrics accumulate element counts; objective diagnostics average
microbatch values across ranks. These reductions should not be confused when
comparing models with different target lengths.

## 11. Rollout validation, checkpoint authority, and test

Formal evaluation uses trajectories from S to T. The evaluator assigns whole
scenarios to ranks without padding, bypasses DDP forward buffer broadcasts,
and globally reduces float64 squared-error, absolute-error, and element-count
accumulators. Uneven shards and ranks with no assigned scenario remain valid.
Metrics include whole-rollout and final-step normalized/physical errors,
per-channel errors, and horizon curves. ISSM also reports thickness and speed
errors, with speed derived from physical vx/vy.

Checkpoint selection minimizes `whole_rollout_norm_rmse` on validation.
After training, the best checkpoint is loaded for final test. TC training
objectives do not replace the validation criterion.

Standalone `scripts/evaluate.py` reconstructs model, H/K, solver, normalizer,
and exact splits from the checkpoint. Allowed runtime changes are data-root
relocation, known_steps, evaluation AMP, output location, and device. H/K,
architecture, training, normalization, and split changes are rejected.
Each new S performs inference with true history ending at S−1.

## 12. Artifacts and postprocessing

Each training output directory records `config.json`, ordered
`config_stack.txt`, `split_files.json`, `train.log`, `history.json`, `best.pt`,
and `final_metrics.json`. The checkpoint embeds merged config, normalizer,
split manifest, model/optimizer state, and the best-epoch criterion. The config
stack is the authoritative record of supplied overlay order.

Standalone evaluation writes complete schema-v2 compressed predictions and
companion JSON metadata. They preserve normalized and physical predictions
and targets, scenario/simulation identifiers, history/start information, lead
steps, absolute indices, adapter times, node identity, and mesh/normalizer
fingerprints. A plotting limit only changes plots; it does not truncate saved
arrays or whole-rollout metrics.

`scripts/postprocess_rollout.py` computes reports without loading the model.
It validates scenario sets, node/channel structure, normalizer identity,
target/time alignment, and requested coverage. Modes include full rollouts,
equal lead, common absolute tail, and lead, index, or adapter-time slices.

For ISSM starts 60/90/120 on T240, equal-lead 120 compares absolute periods
60–179/90–209/120–239. Common-tail uses 120–239 for every rollout, holding the
target period fixed while forecast age differs. Keep each NPZ with its
metadata JSON; repeated postprocessing uses these complete arrays.

## 13. Formal experiment dependencies and provenance

Phase 1 scans H1–H8 with canonical K, full architecture, canonical S, and TC0.
Only its selected H* propagates to the architecture factorial, K scan, and TC scan:

| Phase | History | Architecture | K | TC |
| --- | --- | --- | --- | --- |
| 1. History | H1–H8 | Full | ISSM180 / ANUGA64 | Off |
| 2. Architecture | Selected H* | All 16 combinations | Canonical | Off |
| 3. Training horizon | Selected H* | Full | Dataset-specific K scan | Off |
| 4. Temporal consistency | Selected H* | Full | Canonical | TC0–TC5 |

ISSM K values are 30/45/60/75/90/120/150/180; ANUGA values
are 8/16/24/32/40/48/56/64. S remains 60 for ISSM and 8 for ANUGA.
Architecture and K winners are not propagated into other phases. ISSM
known60/90/120 robustness is a separate evaluation-only use of the same final
checkpoint; ANUGA retains known8.

Dataset-scoped overlays live in `configs/ablations/issm/` and
`configs/ablations/anuga/`. Formal files live under
`launchers/{shell,slurm}/{issm,anuga}/01_history`, `02_architecture`,
`03_training_horizon`, and `04_temporal_consistency`. Every experiment has one
standalone launcher containing its stack and environment. Later-phase files
fail until their visible selected-history placeholders are filled.

`old-files/`, `legacy-scripts/`, and `legacy-v2/` are historical provenance only.
They do not define current configurations or results. Current implementation
and new evidence take precedence over historical documents. See the
[docs index](README.md) and [refactor record](config_evaluation_refactor_20261004.md).
