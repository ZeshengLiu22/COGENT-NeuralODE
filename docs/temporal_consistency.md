# Temporal consistency: current TC0–TC5 equations

This document follows `training/losses.py` and its call in `training/trainer.py`.
NODE2 retains the complete-series state MSE and can add one TC formulation to
the same forward prediction:

\[
L_{\mathrm{total}}
=L_{\mathrm{state}}+\lambda_{\mathrm{TC}}L_{\mathrm{TC}}.
\]

There is no automatic magnitude matching, EMA balancing, or hidden weight 0.3.
The configured outer weight is applied once by the trainer. Every supplied
enabled overlay uses weight 1.0 and MSE. Architecture, rollout evaluation, and
the checkpoint criterion are unchanged; TC requires no additional model forward.

## Tensors, scaling, and reduction operator

Write \(\hat Y,Y\in\mathbb R^{N_\Sigma\times L\times F}\), where \(N_\Sigma\)
is total nodes across a PyG microbatch, \(L=k_{\mathrm{eff}}\), and F is the
number of state channels. These are normalized model predictions/targets
multiplied by `training.loss_scale_factor` before calling either loss:
ISSM 100, ANUGA 1. The base objective is

\[
L_{\mathrm{state}}=\frac{1}{N_\Sigma L F}
\sum_{n=1}^{N_\Sigma}\sum_{t=0}^{L-1}\sum_{c=1}^{F}
(\hat Y_{ntc}-Y_{ntc})^2.
\]

For any nonempty component error tensor e, define

\[
P_{\mathrm{mse}}(e)=\operatorname{mean}_{\text{all elements}}e^2,
\]

or, with `penalty: rmse`,

\[
P_{\mathrm{rmse}}(e)=
\sqrt{\operatorname{mean}_{\text{all elements}}(e^2)+\epsilon}
-\sqrt{\epsilon},\qquad\epsilon>0.
\]

The optional RMSE defaults to ε=1e-8. It is zero with finite zero gradient at
perfect prediction; it differs from exact RMSE near zero. The square root is
applied after elementwise mean, separately for each component/lag. It is not
an average of nodewise or graphwise RMSE values.

All temporal operations use dimension 1. Samples with more nodes contribute
more elements; there is no equal-per-graph loss reduction. Equal B controls
scenario series exposure without changing this elementwise reduction.
TC uses only the predicted future block: the observed history anchor is not
prepended to temporal differences.

## Supplied configuration and phase

Each dataset has `configs/ablations/<dataset>/temporal_consistency/tc0.yaml`
through `tc5.yaml`. Their settings live in `training.temporal_consistency`:

| Overlay | Mode | Enabled | Outer weight | Components |
| --- | --- | --- | ---: | --- |
| tc0 | `none` | false | 1.0, unused | Zero |
| tc1 | `adjacent_increment` | true | 1.0 | Adjacent increments |
| tc2 | `random_pair_increment` | true | 1.0 | One sampled pair per graph by default |
| tc3 | `multiscale_rate` | true | 1.0 | Rate lags 1,3,6,12 |
| tc4 | `rate_curvature` | true | 1.0 | Rate lags 1,3,6,12 plus curvature |
| tc5 | `hybrid` | true | 1.0 | Adjacent plus rate lags 3,6,12 plus curvature |

All component weights in the supplied T4/T5 overlays are 1.0. Their components
are summed without dividing by the number of components. Consequently the
same outer weight does not imply equal regularization magnitude across modes.

Phase 4 fixes selected H* from Phase 1, canonical full architecture,
K180/known60 for ISSM or K64/known8 for ANUGA, and the protocol's fixed B.
It does not inherit the architecture or K-scan winners. The effective horizon
is still sampled independently, and short horizons use the valid terms below.

## T0: off

\[
L_{\mathrm{TC}}=0.
\]

With `enabled: false`, dispatch returns zero total and components. Zero terms
are represented by `y_pred.sum()*0.0` to remain attached to the prediction
graph. An enabled configuration with `mode: none` is invalid.

## T1: adjacent increment

For \(t=0,\ldots,L-2\), form

\[
\Delta\hat Y_t=\hat Y_{t+1}-\hat Y_t,\qquad
\Delta Y_t=Y_{t+1}-Y_t,
\]

\[
L_{\mathrm{adj}}=
P\big(\Delta\hat Y-\Delta Y\big),\qquad
L_{\mathrm{TC}}=L_{\mathrm{adj}}.
\]

The error tensor is `[N_sum,L-1,F]`. P averages nodes, adjacent future pairs,
and channels. No elapsed-time division is applied. If L<2, the term is zero.

## T2: random-pair increment

Eligible pairs are

\[
\mathcal C=\{(i,j):0\le i<j<L,\quad
\ell_{\min}\le j-i,\quad j-i\le\ell_{\max}\text{ if supplied}\}.
\]

`random_pair.min_lag` defaults to 1; `max_lag: null` has no upper restriction.
Each graph g independently selects \(m=\min(M,|\mathcal C|)\) distinct pairs
without replacement, where M=`num_pairs` (default 1). All nodes of graph g use
that graph's pair set \(\mathcal P_g\). Different graphs can coincidentally
select the same pair. For node n belonging to graph g and pair \((i,j)\),

\[
e_{n,(i,j),c}=
(\hat Y_{njc}-\hat Y_{nic})-(Y_{njc}-Y_{nic}),
\]

\[
L_{\mathrm{TC}}=L_{\mathrm{pair}}=P(e).
\]

For MSE, this is explicitly

\[
L_{\mathrm{pair}}=
\frac{1}{N_\Sigma m F}
\sum_{n=1}^{N_\Sigma}
\sum_{(i,j)\in\mathcal P_{g(n)}}\sum_{c=1}^{F}e_{n,(i,j),c}^2.
\]

The intermediate error shape is `[N_sum,m,F]`. There is no elapsed-time division
and no intermediate mean over graph losses. Requesting more pairs than exist
uses all eligible pairs once; L<2 or an empty candidate set yields zero.

The implementation creates independent random priorities for each graph's
eligible pairs and takes top-k. The trainer supplies `batch.batch` and a
dedicated `torch.Generator` on the prediction device, seeded with top-level
`seed + rank` (base seed defaults to 42). Thus TC pair draws do not consume the
global Torch RNG used by horizon sampling or dropout. TC-disabled training
does not consume this generator. Pair reproducibility assumes the same rank
layout and batch order; changing world size need not preserve pair draws.

Direct helper calls without `node_batch` treat all nodes as one graph.
Direct calls without a generator use global Torch RNG. The trainer supplies
both, so those helper defaults do not define formal training behavior.

## T3: multiscale rate

Let τ be the retained future solver time grid. At lag d,

\[
R_t^{(d)}(Y)=\frac{Y_{t+d}-Y_t}{\tau_{t+d}-\tau_t},
\qquad t=0,\ldots,L-d-1,
\]

\[
L_d=P\left(R^{(d)}(\hat Y)-R^{(d)}(Y)\right).
\]

Each lag error has shape `[N_sum,L-d,F]`; P averages all its elements.
Let \(\mathcal D_v=\{d:d<L,\ w_d>0\}\). Then

\[
L_{\mathrm{rate}}=
\frac{\sum_{d\in\mathcal D_v}w_dL_d}
{\sum_{d\in\mathcal D_v}w_d},
\qquad L_{\mathrm{TC}}=L_{\mathrm{rate}}.
\]

Supplied T3 uses lags 1,3,6,12 with equal weights. Optional `rate.lag_weights`
must have the same length as `rate.lags` and be finite, nonnegative, with at
least one positive weight. Invalid-for-this-horizon lags are skipped and valid
weights are renormalized. For L=8, lags 1,3,6 contribute equally; lag 12 drops
out. This is a weighted average of separate lag penalties, not a pooled
average favoring lags with more available time pairs. No valid lag yields zero.

## T4: rate plus curvature

The nonuniform-grid second derivative at an interior future index is

\[
A_t(Y)=
\frac{2}{\tau_{t+1}-\tau_{t-1}}
\left[
\frac{Y_{t+1}-Y_t}{\tau_{t+1}-\tau_t}
-\frac{Y_t-Y_{t-1}}{\tau_t-\tau_{t-1}}
\right],\qquad t=1,\ldots,L-2.
\]

\[
L_{\mathrm{curvature}}=P(A(\hat Y)-A(Y)),
\]

\[
L_{\mathrm{TC}}=
w_rL_{\mathrm{rate}}+w_cL_{\mathrm{curvature}}.
\]

Curvature error has shape `[N_sum,L-2,F]`, with P averaging nodes, interior
times, and channels. It matches target curvature; it does not force predicted
curvature itself toward zero. Curvature is zero for L<3. T4 rate uses the T3
lags 1,3,6,12 and valid-lag renormalization.

Weights are `rate_curvature.rate_weight` and
`rate_curvature.curvature_weight`, both default 1.0. The component sum is not
renormalized when one term becomes unavailable at a short horizon.

## T5: hybrid

\[
L_{\mathrm{TC}}=
w_aL_{\mathrm{adj}}+
w_rL_{\mathrm{rate}}+
w_cL_{\mathrm{curvature}}.
\]

T5 uses rate lags 3,6,12, while T1's adjacent increment already supplies the
local increment component. This choice keeps the supplied hybrid's rate
component focused on longer intervals. Valid-lag renormalization is unchanged.
Weights are `hybrid.adjacent_weight`, `hybrid.rate_weight`, and
`hybrid.curvature_weight`, each default 1.0. Components are added directly;
the outer λ_TC multiplies this sum once.

## Time coordinates, validation, and edge cases

Rate and curvature use `batch.t_future` directly. These times are relative
to the anchor and divided by the trajectory's median positive time spacing
by the dataset. They are not raw physical time, and are not divided by the
model's separate relative-time scale 180/65. T1/T2 use increments without
any time denominator.

Time tensors can be [L], [1,L], or [Q,L] for Q graphs. When a rate/curvature
term is available, times must be finite, strictly increasing, length L, and
identical across all graph rows. Batched unequal time grids are unsupported.

Missing terms return differentiable zero: no adjacent/pair differences at
L=1, no curvature below L=3, and no rate lag with d<L. The helpers return early
when no term is available and do not require a time grid solely for a zero
term. Shape checks otherwise require matching rank-three prediction/target
arrays. Configuration validation runs at trainer initialization. Unknown
modes, invalid penalties, negative/nonfinite weights, invalid pair counts/lags,
and nonpositive epsilon when penalty is RMSE are rejected. MSE permits epsilon zero. Unused mode subsections left by a
recursive overlay merge do not change the selected formulation.

## Objective and prediction logs

| Metric | Meaning |
| --- | --- |
| `train_norm_mse` | Unscaled normalized prediction MSE |
| `train_phys_rmse` | Physical prediction RMSE |
| `train_state_objective` | State MSE after loss-space scaling |
| `train_tc_raw` | Selected temporal objective before outer weight |
| `train_tc_weighted` | λ_TC times raw TC |
| `train_total_objective`, `train_loss` | Optimized state plus weighted TC, before accumulation division |
| `train_tc_adjacent`, `train_tc_random_pair`, `train_tc_rate`, `train_tc_curvature` | Raw components; unused components zero |

Objective/component diagnostics are averaged over microbatches across DDP
ranks, before gradient-accumulation division. Prediction error metrics retain
element-count reductions. The historical `train_loss` alias in the first TC
implementation referred to unscaled normalized MSE; current `train_loss`
means total optimization objective. Use `train_norm_mse` for normalized
prediction-error comparisons. Validation/test still select
`whole_rollout_norm_rmse`, not the training TC objective.
