# Historical launcher archive

Historical launch scripts retained for provenance/reference only.
Do not use these scripts for new formal experiments.

The three retired launcher directories preserve their original file contents.
The retired shared `scripts/sweep_config.sh` and redundant top-level NODE2
wrappers are retained here as well. Their paths and configuration references
are historical and are not maintained or validated as active infrastructure.

Use `launchers/shell/{issm,anuga}/` or `launchers/slurm/{issm,anuga}/` for the
four current formal phases. Each experiment has its own standalone launcher.
Only selected Phase-1 history propagates into Phases 2–4.

Exclude this directory, `old-files/`, and `legacy-v2/` from active launcher
syntax/provenance and stale-reference audits.
