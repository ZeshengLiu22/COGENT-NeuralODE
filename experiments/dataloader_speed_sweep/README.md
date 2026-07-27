# ISSM Dataloader Speed Sweep

Disposable scripts for timing ISSM NODE2 upgrade-v1 training with different
PyTorch/PyG dataloader settings. These files are intentionally isolated from
the main training entrypoint so the experiment folder can be removed later.

By default the sweep uses only `./data/ISSM/PIG_5000`. It still merges
`configs/issm.yaml`, but the generated per-combo override replaces the broader
`PIG_data` path with the PIG-5000 subset.

The sweep defaults to formal-like validation (`full_rollout_on_val: true`) and
does not cap the sampled training horizon unless `--train-horizon-max` is
provided. Use `--skip-full-rollout-on-val` only for train-loader-only timing.

The main training defaults now enable the shared horizon curriculum. For timing
comparisons where the cap should be fixed from epoch 1, pass
`--horizon-curriculum off` to `run_sweep.py`; pass `--horizon-curriculum on` to
force the curriculum on when an override config disabled it.
