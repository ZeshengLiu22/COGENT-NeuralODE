# ISSM dataloader speed sweep

`run_sweep.py` times training-window loader settings with the canonical
`default → datasets/issm → protocols/issm/main → models/node2 → extra overrides`
stack. A generated override records the requested benchmark duration, explicit
data location, and worker settings. The default data is `./data/ISSM/PIG_5000`.
Validation reads complete trajectories from `evaluation.known_steps` to the end.
The timing entrypoint omits the final test and records rollout validation time
separately from training windows per second.

```bash
python experiments/dataloader_speed_sweep/run_sweep.py --max-combos 1 --dry-run
```

Use `--train-horizon-max` only for an explicitly capped timing run. Use
`--horizon-curriculum off` for a fixed target maximum from epoch one.
Every timed run saves its merged `config.json` and ordered `config_stack.txt`.
The measured worker settings are maintained in `configs/runtime/fast.yaml`;
that file contains only runtime options. Benchmark results are not formal model
selection results. See the [handbook](../../handbook.md) for formal experiments.
