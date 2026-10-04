"""Compare complete saved rollout artifacts without loading a model or source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .eval_artifacts import SCENARIO_FIELDS, summarize_full_rollout_bundle, validate_full_rollout_bundle
from .io import save_json


def load_rollout_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load schema-v2 NPZ/JSON with strict coverage and metadata validation."""
    path = Path(path)
    if not path.name.endswith("_predictions.npz"):
        raise ValueError("Artifact filename must end in _predictions.npz.")
    meta_path = path.with_name(path.name.removesuffix("_predictions.npz") + "_predictions_meta.json")
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != 2 or meta.get("mode") != "full_rollout":
        raise ValueError("Expected a complete schema-v2 full_rollout artifact.")
    channels = meta["channel_names"]
    if len(channels) != len(set(channels)) or len(channels) != bundle["pred_phys"].shape[1]:
        raise ValueError("Channel names must be unique and match prediction columns.")
    infos = meta["scenario_lookup"]
    if [info["scenario_index"] for info in infos] != list(range(len(infos))):
        raise ValueError("Scenario metadata indices must be unique and contiguous.")
    validate_full_rollout_bundle(bundle, scenario_infos=infos, known_steps=int(meta["known_steps"]),
                                 node_counts=[int(info["node_count"]) for info in infos])
    for info in infos:
        mask = bundle["scenario_index"] == info["scenario_index"]
        for key in ("known_steps", "history_len", "mesh_signature", "normalization_signature"):
            if bundle[key][info["scenario_index"]] != info[key]:
                raise ValueError(f"Scenario metadata disagrees with stored {key}.")
        if int(info["history_len"]) != int(meta["history_len"]):
            raise ValueError("Global and scenario history_len metadata disagree.")
    return bundle, meta


def _scenario_mask(bundle: dict[str, np.ndarray], scenario_id: str) -> np.ndarray:
    index = np.flatnonzero(bundle["scenario_id"] == scenario_id)
    if len(index) != 1:
        raise ValueError(f"Scenario ID {scenario_id} is missing or ambiguous.")
    return bundle["scenario_index"] == index[0]


def _scenario_rows(bundle: dict[str, np.ndarray], scenario_id: str) -> np.ndarray:
    rows = np.flatnonzero(_scenario_mask(bundle, scenario_id))
    order = np.lexsort((bundle["node_index"][rows], bundle["trajectory_idx0"][rows]))
    return rows[order]


def validate_comparison(artifacts: list[tuple[dict[str, np.ndarray], dict[str, Any]]]) -> list[str]:
    """Align by stable scenario ID; reject incompatible identities, meshes or truth."""
    if not artifacts:
        raise ValueError("At least one rollout artifact is required.")
    reference, reference_meta = artifacts[0]
    scenario_ids = sorted(str(info["scenario_id"]) for info in reference_meta["scenario_lookup"])
    reference_infos = {str(info["scenario_id"]): info for info in reference_meta["scenario_lookup"]}
    for bundle, meta in artifacts[1:]:
        if meta["dataset_name"] != reference_meta["dataset_name"]:
            raise ValueError("Cannot compare different datasets.")
        if meta["channel_names"] != reference_meta["channel_names"]:
            raise ValueError("Channel names/order do not match across artifacts.")
        infos = {str(info["scenario_id"]): info for info in meta["scenario_lookup"]}
        if sorted(infos) != scenario_ids:
            raise ValueError("Scenario ID sets do not match; comparison never silently drops scenarios.")
        for scenario_id in scenario_ids:
            left, right = reference_infos[scenario_id], infos[scenario_id]
            for key in ("sim_id", "length", "node_count", "mesh_signature", "normalization_signature"):
                if left[key] != right[key]:
                    raise ValueError(f"Scenario {scenario_id}: incompatible {key} (node structure/identity).")
            left_rows = _scenario_rows(reference, scenario_id)
            right_rows = _scenario_rows(bundle, scenario_id)
            common_indices = np.intersect1d(reference["trajectory_idx0"][left_rows], bundle["trajectory_idx0"][right_rows])
            if not common_indices.size:
                raise ValueError(f"Scenario {scenario_id} has no common absolute future interval.")
            left_rows = left_rows[np.isin(reference["trajectory_idx0"][left_rows], common_indices)]
            right_rows = right_rows[np.isin(bundle["trajectory_idx0"][right_rows], common_indices)]
            for key in ("node_index", "trajectory_idx0", "trajectory_time"):
                if not np.array_equal(reference[key][left_rows], bundle[key][right_rows]):
                    raise ValueError(f"Scenario {scenario_id}: {key} alignment mismatch.")
            for key in ("target_phys", "target_norm"):
                if not np.allclose(reference[key][left_rows], bundle[key][right_rows], rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Scenario {scenario_id}: {key} mismatch at common absolute times.")
    return scenario_ids


def compare_rollout_artifacts(
    paths: list[str | Path], *, mode: str = "full", labels: list[str] | None = None,
    lead_steps: int | None = None, lead_start: int = 1, lead_stop: int | None = None,
    absolute_start: int | None = None, absolute_stop: int | None = None,
    time_start: float | None = None, time_stop: float | None = None,
) -> dict[str, Any]:
    """Recompute metrics over full, equal-lead, common-tail or user-selected slices.

    Lead bounds are one-based and inclusive. Absolute/time bounds are half-open.
    Aggregation weights every node/time/channel equally, matching the evaluator.
    """
    if mode not in ("full", "equal-lead", "common-tail", "lead-slice", "absolute-slice", "time-slice"):
        raise ValueError(f"Unknown comparison mode: {mode}")
    if mode == "equal-lead" and (lead_steps is None or lead_steps < 1):
        raise ValueError("equal-lead requires positive lead_steps.")
    if mode == "lead-slice" and (lead_start < 1 or (lead_stop is not None and lead_stop < lead_start)):
        raise ValueError("Invalid inclusive lead-step bounds.")
    if mode == "absolute-slice" and (absolute_start is None and absolute_stop is None):
        raise ValueError("absolute-slice requires at least one absolute bound.")
    if mode == "absolute-slice":
        if absolute_start is not None and absolute_start < 0:
            raise ValueError("Absolute indices must be nonnegative.")
        if absolute_start is not None and absolute_stop is not None and absolute_stop <= absolute_start:
            raise ValueError("Absolute stop must be greater than start.")
    if mode == "time-slice":
        if time_start is None and time_stop is None:
            raise ValueError("time-slice requires at least one physical-time bound.")
        if any(value is not None and not np.isfinite(value) for value in (time_start, time_stop)):
            raise ValueError("Physical-time bounds must be finite.")
        if time_start is not None and time_stop is not None and time_stop <= time_start:
            raise ValueError("Physical-time stop must be greater than start.")
    artifacts = [load_rollout_artifact(path) for path in paths]
    scenario_ids = validate_comparison(artifacts)
    if labels is None:
        labels = [Path(path).name.removesuffix("_predictions.npz") for path in paths]
    if len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per artifact.")
    common = {}
    if mode == "common-tail":
        for scenario_id in scenario_ids:
            indices = [np.unique(bundle["trajectory_idx0"][_scenario_mask(bundle, scenario_id)]) for bundle, _ in artifacts]
            common[scenario_id] = np.arange(max(int(values[0]) for values in indices), min(int(values[-1]) for values in indices) + 1)
            if not common[scenario_id].size:
                raise ValueError(f"Scenario {scenario_id} has an empty common tail.")
    results = []
    for path, label, (bundle, meta) in zip(paths, labels, artifacts):
        mask = np.ones(len(bundle["scenario_index"]), dtype=bool)
        if mode == "equal-lead":
            mask &= bundle["horizon_idx1"] <= lead_steps
        elif mode == "lead-slice":
            mask &= bundle["horizon_idx1"] >= lead_start
            if lead_stop is not None:
                mask &= bundle["horizon_idx1"] <= lead_stop
        elif mode == "common-tail":
            mask[:] = False
            for scenario_id in scenario_ids:
                mask |= _scenario_mask(bundle, scenario_id) & np.isin(bundle["trajectory_idx0"], common[scenario_id])
        elif mode in ("absolute-slice", "time-slice"):
            values = bundle["trajectory_idx0"] if mode == "absolute-slice" else bundle["trajectory_time"]
            lower, upper = (absolute_start, absolute_stop) if mode == "absolute-slice" else (time_start, time_stop)
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values < upper
        coverage = []
        for scenario_id in scenario_ids:
            scenario_mask = _scenario_mask(bundle, scenario_id)
            if mode == "absolute-slice":
                available = bundle["trajectory_idx0"][scenario_mask]
                if ((absolute_start is not None and absolute_start < int(available.min()))
                        or (absolute_stop is not None and absolute_stop > int(available.max()) + 1)):
                    raise ValueError(f"{label}: scenario {scenario_id} does not cover the requested absolute slice.")
            if mode == "time-slice" and time_start is not None:
                # The observed anchor is the last omitted state. A lower bound
                # between that time and the first forecast time omits no state.
                anchor_time = float(bundle["history_end_time"][scenario_mask][0])
                if time_start <= anchor_time:
                    raise ValueError(f"{label}: scenario {scenario_id} does not cover the requested physical-time slice.")
            rows = mask & scenario_mask
            if not rows.any():
                raise ValueError(f"{label}: slice contains no rows for scenario {scenario_id}.")
            absolute = np.unique(bundle["trajectory_idx0"][rows])
            leads = np.unique(bundle["horizon_idx1"][rows])
            if mode == "equal-lead" and len(leads) != lead_steps:
                raise ValueError(f"{label}: scenario {scenario_id} has fewer than {lead_steps} lead steps.")
            if mode == "lead-slice" and lead_stop is not None and len(leads) != lead_stop - lead_start + 1:
                raise ValueError(f"{label}: scenario {scenario_id} does not cover the requested lead slice.")
            coverage.append({"scenario_id": scenario_id, "absolute_indices": absolute.tolist(),
                             "actual_times": np.unique(bundle["trajectory_time"][rows]).tolist(),
                             "lead_steps": leads.tolist(), "node_count": int(np.unique(bundle["node_index"][rows]).size),
                             "row_count": int(rows.sum())})
        sliced = {key: values if key in SCENARIO_FIELDS else values[mask] for key, values in bundle.items()}
        summary = summarize_full_rollout_bundle(sliced, meta["channel_names"], summary_method=label)
        results.append({"label": label, "artifact": str(path), "known_steps": meta["known_steps"],
                        "history_len": meta["history_len"], "scenarios": coverage,
                        "metrics": summary["metrics"], "metric_table": summary["metric_table"],
                        "leadtime_metrics": summary["leadtime_metrics"],
                        "absolute_time_metrics": summary["trajectory_metrics"]})
    if mode in ("absolute-slice", "time-slice") and len(results) > 1:
        reference_coverage = {row["scenario_id"]: row for row in results[0]["scenarios"]}
        for result in results[1:]:
            for row in result["scenarios"]:
                reference = reference_coverage[row["scenario_id"]]
                if (row["absolute_indices"] != reference["absolute_indices"]
                        or row["actual_times"] != reference["actual_times"]):
                    raise ValueError(f"{result['label']}: selected absolute/time interval differs for scenario {row['scenario_id']}; use common-tail for the shared interval.")
    return {"mode": mode, "bounds": {"lead_steps": lead_steps, "lead_start": lead_start,
                                     "lead_stop": lead_stop, "absolute_start": absolute_start,
                                     "absolute_stop": absolute_stop, "time_start": time_start, "time_stop": time_stop},
            "dataset_name": artifacts[0][1]["dataset_name"], "channel_names": artifacts[0][1]["channel_names"],
            "aggregation": "Global SSE/count and absolute-error/count over selected elements; final_step is the last selected step per scenario.",
            "results": results}


def save_comparison_report(report: dict[str, Any], output_prefix: str | Path) -> dict[str, str]:
    """Write full JSON plus scalar metrics CSV and coordinate-bearing curve CSVs."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {"json": f"{prefix}.json", "csv": f"{prefix}.csv",
             "leadtime_csv": f"{prefix}_leadtime.csv", "absolute_time_csv": f"{prefix}_absolute_time.csv"}
    save_json(paths["json"], report)
    tables = {"csv": [], "leadtime_csv": [], "absolute_time_csv": []}
    for result in report["results"]:
        identity = {"label": result["label"], "known_steps": result["known_steps"], "history_len": result["history_len"]}
        tables["csv"].append({**identity, **{key: value for key, value in result["metrics"].items() if not isinstance(value, list)}})
        tables["leadtime_csv"].extend({**identity, **row} for row in result["leadtime_metrics"])
        tables["absolute_time_csv"].extend({**identity, **row} for row in result["absolute_time_metrics"])
    for name, rows in tables.items():
        with Path(paths[name]).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
