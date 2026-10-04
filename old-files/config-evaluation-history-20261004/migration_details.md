# Historical names removed by the 2026-10-04 migration

This is an archival deletion record. These keys, paths, and behaviors are no
longer supported. Use the [current handbook](../../handbook.md).

The old configuration tree included `base_sample.yaml`, complete
`ISSM_History_Scan/`, `ANUGA_History_Scan/`, `ISSM_Future_Len_Ablation/` bases,
combined `NODE2_Upgrade1_Ablation/` presets, `TemporalConsistency/` overlays,
root `issm.yaml`, `anuga.yaml`, `adcirc.yaml`, `model_node2.yaml`, and
`issm_paper_matched.yaml`. All have been migrated or removed from active configs.

- `window_reference` was removed from enumeration, dataset construction,
  configs, audit assertions, and active documentation. There is no replacement
  shared-anchor mechanism.
- `evaluation.full_rollout_known_steps` was renamed to
  `evaluation.known_steps`.
- `evaluation.full_rollout_on_val`, window-evaluation batch/workers,
  `val_window`, `test_window`, and `Evaluator.evaluate_loader()` were removed.
- `scripts/run_full_rollout.py` was consolidated into `scripts/evaluate.py`;
  the former fixed-window `evaluate.py` implementation was deleted.
- `eval_window_tmux.sh`, duplicate full-rollout/ANUGA evaluation wrappers,
  and old combined architecture/H6 future wrappers were removed.
- `BASE_CONFIG` / `LOADER_CONFIG` and paper-matched basename branching were
  removed from active launchers.
- Standard ISSM `PIG_data` selection was replaced with `PIG_5000`; dataset
  selection was removed from performance overlays.
- The old `upgrade_off` description did not denote an LSTM baseline. The new
  explicit Transformer/LSTM axis removes that ambiguity.

Exact deleted, moved, added, and modified paths are in `file_inventory.json`.
