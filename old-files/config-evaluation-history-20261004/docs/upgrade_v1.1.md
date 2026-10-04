# Upgrade v1.1: Horizon Curriculum

Upgrade v1.1 adds an epoch-wise training horizon curriculum to the shared
trainer. It is independent of the NODE2 architecture changes from upgrade v1
and works for both ANUGA and ISSM because both datasets use the same
`Trainer.train_epoch` path.

## Motivation

Long NODE rollouts can be unstable early in training. The curriculum keeps the
effective training horizon shorter at the beginning, then gradually increases
the maximum sampled horizon toward the experiment's target cap.

This is especially useful for ISSM long-rollout experiments where the formal
evaluation may be longer than the supervised fixed-window `dataset.future_len`.

## Config

The base configs enable the curriculum by default:

```yaml
training:
  train_horizon_mode: biased_long_horizon
  train_horizon_min: 24
  train_horizon_max: 120
  train_horizon_curriculum:
    enabled: true
    epochs: 120
    warmup_fractions: [0.40, 0.55, 0.70, 0.85]
```

`train_horizon_max` is still the target cap for the run. If it is `null`, the
target cap is `dataset.future_len`.

## Schedule

For each epoch, the trainer resolves an epoch-local maximum horizon:

```text
current_max = train_horizon_min
              + fraction * (target_train_horizon_max - train_horizon_min)
```

The result is rounded to the nearest integer and clamped to:

```text
train_horizon_min <= current_max <= target_train_horizon_max
```

With the default four fractions, the first 120 epochs are split into four
equal stages. After epoch 120, `current_max` stays at the target cap.

For ISSM with `train_horizon_min: 24`, the proposed ablation targets produce:

| Target max | Epoch 1-30 | Epoch 31-60 | Epoch 61-90 | Epoch 91-120 | Epoch 121-300 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 30 | 33 | 35 | 38 | 40 |
| 60 | 38 | 44 | 49 | 55 | 60 |
| 80 | 46 | 55 | 63 | 72 | 80 |
| 100 | 54 | 66 | 77 | 89 | 100 |
| 120 | 62 | 77 | 91 | 106 | 120 |

For ANUGA defaults with `train_horizon_min: 8` and target `64`, the stage caps
are:

```text
30, 39, 47, 56, 64
```

## Sampling

The curriculum only changes the maximum horizon available to the existing
sampler. Within each training step:

1. Resolve `current_train_horizon_max` for the epoch.
2. Sample `k_eff` from `train_horizon_min..current_train_horizon_max` using
   `training.train_horizon_mode`.
3. Truncate `force_future`, `t_future`, `y_future`, and future metadata to
   `k_eff`.
4. Run the model and compute loss on the truncated rollout.

Fixed-window validation and full-rollout validation do not use the curriculum
cap. They keep their configured evaluation horizons.

## CLI Override

Direct training runs can override only the enable flag:

```bash
python scripts/train.py ... --horizon-curriculum off
python scripts/train.py ... --horizon-curriculum on
```

The shared tmux launchers pass extra trainer arguments through `TRAIN_ARGS`:

```bash
TRAIN_ARGS="--horizon-curriculum off" ./train_anuga_node2.sh
TRAIN_ARGS="--horizon-curriculum off" ./train_issm_node2.sh
```

The formal ISSM sbatch script also accepts `TRAIN_ARGS`.

## Ablation Guidance

For ISSM training-horizon ablations, keep the fixed-window dataset length
constant and vary only the target training cap:

```yaml
dataset:
  future_len: 120

training:
  train_horizon_min: 24
  train_horizon_max: 40  # or 60, 80, 100, 120
```

This isolates the effect of maximum backpropagated rollout length. Changing
`dataset.future_len` at the same time would also change sample construction,
the number of valid windows, validation-window length, and the maximum possible
training horizon.

## Logging

The trainer records the epoch-local and target caps in each training metrics
record:

```json
{
  "train_horizon_min": 24.0,
  "train_horizon_max": 62.0,
  "train_horizon_target_max": 120.0
}
```

Epoch logs show the same relationship compactly:

```text
train_horizon_max=62/120
```

## Backward Compatibility

To recover fixed target-cap sampling from epoch 1, disable the curriculum:

```yaml
training:
  train_horizon_curriculum:
    enabled: false
```

or pass:

```bash
--horizon-curriculum off
```
