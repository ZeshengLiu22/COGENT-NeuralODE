# Horizon curriculum and variable-length training series

The horizon curriculum changes the maximum available sampled rollout length
during training. It is independent of the [four NODE2 architecture factors](upgrade_v1.md)
and of the per-scenario epoch series budget. The current implementation is
`training/horizon_sampling.py` and `Trainer.train_epoch` in `training/trainer.py`.

## B counts series; k_eff determines prediction length

\[
K=\texttt{dataset.future_len},\qquad k_{\mathrm{eff}}\le K.
\]

K is the maximum future block stored for a selected scenario/anchor. The
trainer later truncates it to k_eff and predicts every retained future state.

\[
B=\texttt{dataset.train_series_per_scenario_per_epoch}
\]

is how many training series each simulation contributes per epoch. Formal B is
60 for ISSM and 9 for ANUGA. Natural H/K-specific anchors are sampled to meet
this count, with replacement only for the extra draws needed when N<B.
This fixes series exposure without fixing k_eff, target timestep count,
series ending time, or total integration work.

A repeated anchor can be trained with different sampled horizons on different
optimizer steps. If gradient accumulation is enabled, all microbatches in one
optimizer group share the group's k_eff. The default accumulation setting is 1.

## Target horizon and configuration

```yaml
training:
  train_horizon_mode: biased_long_horizon
  train_horizon_max: null
  train_horizon_curriculum:
    enabled: true
    epochs: 120
    warmup_fractions: [0.40, 0.55, 0.70, 0.85]
```

The protocol supplies `train_horizon_min: 24` for ISSM or `8` for ANUGA.
The target upper bound is

\[
K_{\max}^{target}=
\begin{cases}
K,&\texttt{train_horizon_max=null},\\
\texttt{train_horizon_max},&\text{otherwise}.
\end{cases}
\]

Require \(1\le K_{\min}\le K_{\max}^{target}\le K\). All supplied K overlays
change `dataset.future_len` and keep the null maximum, so their target maximum
tracks K. ANUGA K8 has K_min=K_target=8 and is valid throughout training.

## Exact epoch staircase

Epoch indices are one-based. Let E be curriculum epochs, m the number of
fractions, and \(f_0,\ldots,f_{m-1}\) those fractions. For \(1\le e\le E\),

\[
j(e)=\min\left(\left\lfloor\frac{(e-1)m}{E}\right\rfloor,m-1\right),
\qquad \alpha_e=f_{j(e)}.
\]

The unrounded cap has the intended form

\[
K_e^{raw}=K_{\min}
+\alpha_e(K_{\max}^{target}-K_{\min}).
\]

The implementation uses round-half-up on the nonnegative span, then clamps:

\[
K_e=\max\left(K_{\min},
\min\left(K_{\max}^{target},
K_{\min}+\left\lfloor
\alpha_e(K_{\max}^{target}-K_{\min})+0.5
\right\rfloor\right)\right).
\]

This is `math.floor(span*fraction + 0.5)`, not Python's ties-to-even `round`.
After epoch E, K_e equals the target. The target is also used immediately when
curriculum is disabled, E≤0, or target equals minimum. Epoch<1 and a target
below minimum are rejected. Active fractions must be nonempty and in (0,1];
their listed order defines stages.

For E=120 and four fractions, epochs 1–30/31–60/61–90/91–120 use the four
fractions respectively, then epochs 121 onward use the target:

| Protocol and K | 1–30 | 31–60 | 61–90 | 91–120 | 121 onward |
| --- | ---: | ---: | ---: | ---: | ---: |
| ISSM K30, min 24 | 26 | 27 | 28 | 29 | 30 |
| ISSM K60, min 24 | 38 | 44 | 49 | 55 | 60 |
| ISSM K120, min 24 | 62 | 77 | 91 | 106 | 120 |
| ISSM K180, min 24 | 86 | 110 | 133 | 157 | 180 |
| ANUGA K8, min 8 | 8 | 8 | 8 | 8 | 8 |
| ANUGA K32, min 8 | 18 | 21 | 25 | 28 | 32 |
| ANUGA K64, min 8 | 30 | 39 | 47 | 56 | 64 |

## Independent effective-horizon sampling

Given the epoch cap,

\[
k_{\mathrm{eff}}\sim p(k\mid K_{\min},K_e).
\]

Both supported distributions use every integer endpoint inclusively:

\[
p_{\mathrm{uniform}}(k)=\frac{1}{K_e-K_{\min}+1},
\]

\[
p_{\mathrm{biased}}(k)=\frac{k}{\sum_{j=K_{\min}}^{K_e}j},
\qquad K_{\min}\le k\le K_e.
\]

Their configuration names are `uniform_random` and `biased_long_horizon`.
The formal default is the latter. There is no fixed rollout length per anchor.

The series sampler uses an epoch-seeded NumPy generator. Horizon sampling uses
Torch and is synchronized across DDP ranks: rank 0 samples once at the beginning
of an optimization/accumulation group and broadcasts its integer. Changing
the epoch budget does not replace this sampler. TC random-pair sampling uses
a separate rank-local generator and does not consume horizon/model RNG.

## Trainer execution and full-series supervision

1. The dataset selects scenario/anchor series for the epoch.
2. Each batch initially contains its maximum K future forcing/target values.
3. The trainer resolves K_e and samples synchronized k_eff.
4. `_truncate_future_horizon` trims `force_future`, `y_future`, and optional
   `target_mask` along dimension 1. It also trims `t_future`, `future_idx`, and
   `future_time` consistently, supporting graph metadata shaped [Q,K] or [K].
5. NODE2 integrates at [0,τ1,…,τ_k_eff], drops its initial latent, and decodes
   the complete future series.
6. State MSE and optional TC consume matching `[N_total,k_eff,F_state]` arrays.
7. The loss is divided by the actual accumulation-group size; the final partial
   group therefore retains its intended gradient scale.

The saved maximum K block and the actual k_eff prediction have distinct roles.
No step selects only the last future state. Training loss averages all retained
state times, nodes, and channels.

## Evaluation and experiment design

Validation/test predict from S to trajectory end and do not use a sampled
horizon or curriculum cap. For T240 at known60, an ISSM K30 checkpoint produces
180 future predictions. Checkpoint H/K and the fixed relative-time scale remain
authoritative; S is the supported rollout-start override.

In Phase 3, hold selected H*, canonical full architecture, canonical S, and TC0.
ISSM scans K30/45/60/75/90/120/150/180; ANUGA scans
K8/16/24/32/40/48/56/64. B remains 60 or 9 respectively, giving fixed epoch
series counts despite differing natural-anchor counts. The epoch-based
optimizer/scheduler schedule is retained. Phase 4 returns to canonical K and
full architecture; it does not inherit the K-scan winner.

Each Phase-3 launcher is independent and requires its visible selected-history
placeholder to be set after Phase 1. See the [operating handbook](../handbook.md).

## Logging and an explicit curriculum override

The trainer records `train_horizon_min`, `train_horizon_max` (current cap),
and `train_horizon_target_max` for each epoch. The text log shows
`train_horizon_max=current/target`. Those caps are not a record of every
sampled k_eff.

For a deliberate ad-hoc curriculum comparison, `scripts/train.py` accepts
`--horizon-curriculum off` or `on`. The merged saved configuration reflects
this override. Formal launchers retain the specified curriculum; a new
curriculum comparison is separate from the four requested phases.
