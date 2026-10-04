# Final cleanup validation — 2026-10-04

This record covers the cleanup following `f96cae3`, including per-graph T2
sampling, an independent `seed + rank` generator, zero-valued stabilized RMSE,
the `train_loss` reporting migration, launcher overlay separation, and the H5
gradient-clip configuration correction. These checks were executed locally;
they are not a claim of a GitHub Actions run or a real-data accuracy result.

## Environment and commands

Python 3.11.14; PyTorch 2.8.0+cu128; CPU execution (CUDA unavailable).

```bash
export PYTHON_BIN=/work2/09575/zeshengliu/conda_envs/torchpyg-cu128-cge/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m compileall -q datasets models training scripts utils tests
GLOO_SOCKET_IFNAME=lo "$PYTHON_BIN" tests/check_temporal_ddp.py
git diff --check
```

Full unittest result:

```text
Ran 95 tests in 13.627s

OK
```

Compilation and diff whitespace checks succeeded. `bash -n` also passed for all
nine modified launchers: the two shared training scripts and the seven formal
history/future/architecture-ablation helpers.

## Covered behavior

- All existing foundation, dataset, model, evaluation, checkpoint, and temporal
  regressions pass in the full suite.
- T2 draws independent pair sets per graph, shares each set within that graph,
  samples without replacement, and retains node-element weighting for unequal
  graph sizes. Seeded private generators reproduce draws without consuming
  the global model RNG.
- Perfect-prediction RMSE is zero with finite zero gradients. RMSE rejects a
  zero, negative, or nonfinite stabilization epsilon.
- Trainer regression tests compare actual dropout masks and RNG state between
  baseline and T2, verify rank-specific generator seeds, and check objective
  accounting plus gradient accumulation, including a partial final group.
- `train_loss` equals the optimization-space `train_total_objective`. Existing
  normalized and physical prediction metrics retain their reductions. See the
  [logging migration](temporal_consistency.md#logging-migration-after-f96cae3).
- Actual Bash command construction is exercised with a recording Python stub;
  no training, tmux, or scheduler task starts in launcher tests. Merged configs
  retain paper-matched H1/K239/time-scale239/known1 and retain the formal loader
  overlay when TC is added. All three scan families vary only their designated
  history/future length after normalizing that field.
- The real NODE2, ANUGA/ISSM dataset adapters, normalizer, trainer, and evaluator
  run on synthetic tiny trajectories for six combinations: both datasets with
  TC off, random pair, and multiscale rate. Each case performs one training
  epoch, updates parameters, saves a checkpoint, reloads it, verifies identical
  predictions after reload, and produces finite history/evaluation metrics.

## Two-process DDP check

The standalone `tests/check_temporal_ddp.py` requires local TCP sockets and is
run separately from ordinary unittest discovery. Both CPU/Gloo ranks passed:

```text
rank 0: graph-batched T2, private seed 77, dropout RNG, objective all-reduce, one forward PASS
rank 1: graph-batched T2, private seed 78, dropout RNG, objective all-reduce, one forward PASS
```

This checks unequal graph sizes, unchanged dropout RNG relative to baseline,
identical reduced diagnostics across ranks, and exactly one forward per batch.

## Scope

The model, dataset adapters, horizon sampling, evaluation, and checkpoint
selection were not changed. No full training sweep was launched. Solver-step
convergence on a newly trained clean real-data checkpoint remains an experiment
to run after that checkpoint exists; the existing diagnostic implementation is
unchanged. Historical documents are explicitly labeled and current launch/TC
instructions are in the [shell handbook](../handbook.md) and
[temporal-consistency guide](temporal_consistency.md).
