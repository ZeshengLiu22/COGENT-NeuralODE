# Temporal consistency training

NODE2 always retains its rollout state MSE. A run can add exactly one configured
temporal-consistency (TC) mode to the same forward prediction:

$$
L_{\mathrm{total}} = L_{\mathrm{state}} + \lambda_{\mathrm{TC}} L_{\mathrm{TC}}.
$$

There is no automatic magnitude matching or EMA balancing. TC and state loss
use the same normalized, `training.loss_scale_factor`-scaled prediction and
target tensors, shaped `[N_total_nodes, K, F_state]`. Every temporal difference
uses dimension 1. The model architecture, dataset, validation/test metrics, and
checkpoint criterion are unchanged; computing TC adds no model forward.

## Modes and config

All settings live under `training.temporal_consistency`. Append the chosen
overlay after model/selection overlays and before the runtime overlay. Run this
phase only after choosing H, architecture, and K; see the [handbook](../handbook.md).

| Overlay in `configs/ablations/temporal_consistency/` | Mode | Temporal objective |
| --- | --- | --- |
| `tc0_off.yaml` | `none`, disabled | Zero; state MSE only |
| `tc1_adjacent_increment.yaml` | `adjacent_increment` | Match adjacent predicted and true increments |
| `tc2_random_pair_increment.yaml` | `random_pair_increment` | Match increments between independently sampled time pairs for each graph |
| `tc3_multiscale_rate.yaml` | `multiscale_rate` | Weighted mean of rate penalties over lags `[1, 3, 6, 12]` |
| `tc4_rate_curvature.yaml` | `rate_curvature` | `rate_weight * rate + curvature_weight * curvature` |
| `tc5_hybrid.yaml` | `hybrid` | `adjacent_weight * adjacent + rate_weight * rate + curvature_weight * curvature`; rate lags `[3, 6, 12]` |

The enabled overlays use outer `weight: 1.0`, `penalty: mse`, and component
weights of 1.0. T4/T5 are raw component sums; they are not divided by the number
of components. Their regularization magnitude can therefore differ from T1–T3
at the same outer weight. Use the logged objectives when interpreting results.

```yaml
training:
  temporal_consistency:
    enabled: true
    mode: random_pair_increment
    weight: 1.0
    penalty: mse               # mse or rmse
    rmse_eps: 1.0e-8
    random_pair:
      num_pairs: 1
      min_lag: 1
      max_lag: null
```

`enabled` defaults to false; `weight` defaults to 1.0. Unknown modes are errors,
and an enabled mode cannot be `none`. Rate modes use `rate.lags` and optional
`rate.lag_weights`; weights must match the lag list. T4 component weights live
in `rate_curvature`, and T5 weights live in `hybrid`. Those component weights
default to 1.0. Unused mode subsections from earlier overlays have no effect.

## Per-graph random pairs and RNG isolation

T2 samples eligible pairs $i<j$ independently for each graph/sample in a PyG
microbatch. All nodes in one graph share its selected pair set; different
graphs make independent draws and may coincidentally select the same pairs.
Pairs are sampled without replacement within each graph. If fewer eligible
pairs exist than `num_pairs`, all eligible pairs are used. The penalty averages
over the selected node/pair/feature errors, retaining node weighting across
different graph sizes. Pair increments are not divided by elapsed time.

The trainer passes the PyG node-to-graph assignment (`batch.batch`) and a
dedicated generator to T2. That generator is seeded with the run's top-level
`seed + rank` (base seed defaults to 42). Pair sampling consumes only this
generator, leaving the global Torch RNG used by model/dropout unchanged. This
supports controlled comparisons for the same seed, rank layout, and batch
order; changing world size does not promise identical pair selections.
TC-disabled runs do not consume pair-sampling RNG. Direct helper calls without
a node-to-graph assignment treat all nodes as one graph; direct calls without
a supplied generator use the global Torch RNG. The trainer supplies both.

## Rates, curvature, and short horizons

Rate and curvature use the solver's existing `t_future` coordinates directly:

$$
R_t^{(d)} = \frac{Y_{t+d}-Y_t}{\tau_{t+d}-\tau_t}, \qquad
A_t = \frac{2}{\tau_{t+1}-\tau_{t-1}}
\left(\frac{Y_{t+1}-Y_t}{\tau_{t+1}-\tau_t}
-\frac{Y_t-Y_{t-1}}{\tau_t-\tau_{t-1}}\right).
$$

Each component penalizes the difference between predicted and true increments,
rates, or curvature. Curvature matching does not drive the predicted curvature
itself to zero. A time grid may have shape `[K]`, `[1,K]`, or `[B,K]`; batched
rows must agree and times must be finite and strictly increasing.

Lags $d\ge K$ are skipped, and remaining lag weights are renormalized. Missing
terms, including increments at K1 and curvature below K3, return differentiable
zero. Thus curriculum truncation can shorten the supervised horizon without
requiring new temporal configs.

## MSE and stable zero-valued RMSE

For component error `e`, MSE is `mean(e ** 2)`. Optional RMSE uses the shifted,
smoothed expression

$$
P_{\mathrm{rmse}}(e) = \sqrt{\operatorname{mean}(e^2)+\epsilon}
- \sqrt{\epsilon}, \qquad \epsilon > 0.
$$

`rmse_eps` must be finite and strictly positive when `penalty: rmse` (MSE
allows zero because it does not use the epsilon). This expression is zero at
perfect prediction and has a finite zero gradient there. It differs from exact
RMSE close to zero. All supplied experiment overlays continue to use MSE.

## Logging migration after `f96cae3`

`train_loss` now aliases `train_total_objective`, the actual optimized objective
before gradient-accumulation division. This is a deliberate reporting change
from `f96cae3`, where `train_loss` aliased unscaled `train_norm_mse`. It applies
both with TC enabled and disabled; even without TC, `loss_scale_factor != 1`
can make the objective differ from normalized prediction MSE. For old/new
prediction-error comparisons, use `train_norm_mse`, whose meaning is unchanged.

| Metric | Meaning |
| --- | --- |
| `train_norm_mse` | Unscaled normalized prediction MSE; existing metric reduction unchanged |
| `train_phys_rmse` | Physical prediction RMSE; existing metric reduction unchanged |
| `train_state_objective` | State MSE after the configured loss scaling |
| `train_tc_raw` | Selected TC objective before the outer weight |
| `train_tc_weighted` | `weight * train_tc_raw` |
| `train_total_objective`, `train_loss` | State objective plus weighted TC |
| `train_tc_adjacent`, `train_tc_random_pair`, `train_tc_rate`, `train_tc_curvature` | Raw component values; unused components are zero |

Objective/component logs are averaged over training microbatches across DDP
ranks, before gradient-accumulation division. No `train_tc_scale` is needed
because no balancer is used. Validation/test metrics and checkpoint selection
continue to measure prediction accuracy, not the TC training objective.
