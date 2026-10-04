# Active documentation

These references describe the current NODE2 implementation and formal ISSM/
ANUGA experiment design. Current code takes precedence over historical notes.

| Document | Purpose |
| --- | --- |
| [continuous_graph_emulator_master_handbook.md](continuous_graph_emulator_master_handbook.md) | Current end-to-end technical reference: adapters, normalization, sampling, model, losses, rollout evaluation, artifacts, and postprocessing |
| [upgrade_v1.md](upgrade_v1.md) | Current NODE2 architecture equations and the 16-combination architecture ablation |
| [upgrade_v1.1.md](upgrade_v1.1.md) | Current horizon curriculum and independent k_eff training design |
| [temporal_consistency.md](temporal_consistency.md) | Current TC0–TC5 implementation, equations, averaging dimensions, and RNG |
| [config_evaluation_refactor_20261004.md](config_evaluation_refactor_20261004.md) | H/K/S and rollout-only refactor record, including the final series-budget and launcher cleanup |
| [Operating handbook](../handbook.md) | Explicit formal phases, launch commands, evaluation, and postprocessing |
| [Configuration reference](../config_setting.md) | Layer ownership, dataset-scoped overlays, field meanings, and evaluation authority |

H is history length, K is maximum training future, and S is rollout start.
B fixes each simulation's epoch series count; k_eff remains independently
sampled. Only selected H* propagates from Phase 1 into Phases 2–4.

`old-files/` and `legacy-scripts/` are historical only.
Their configurations, launcher paths, and deferred-model designs are not
active instructions. Historical launch scripts are retained for provenance/
reference only; do not use them for new formal experiments.

[Current cleanup validation](final_cleanup_validation/results.json) records the
completed checks and current CUDA smoke status. Its `runtime_cache_fix_validation`
section records the CPU/configuration/launcher recheck after the runtime and
account fixes; earlier audit entries retain their original scope.

Validation logs and JSON evidence are records of the exact run that generated
them. Files under `refactor_validation/` record the earlier rollout-only
refactor; they do not by themselves validate subsequent changes. A CPU pass
does not establish CUDA training validation, and smoke tests are not formal
scientific results.
