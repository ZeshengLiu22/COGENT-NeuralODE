#!/usr/bin/env python3
"""Read-only CPU audit of inputs, rollout semantics, and natural anchors and fixed training-series exposure.

Example:
    python scripts/audit_foundation.py --data-root /path/to/data \
        --output docs/foundation_audit.json

Only rainfall, mesh size, and time arrays are read from ANUGA archives. ISSM
MAT structs are read one at a time, with no training or model evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import numpy as np
import torch

from datasets.factory import build_splits
from datasets.issm_dataset import ISSMDataset, _load_container
from datasets.normalization import FeatureNormalizer, _RunningStats
from datasets.split_utils import parse_issm_rate_from_filename
from datasets.window_utils import WindowMetadata, enumerate_window_end_indices, sample_training_series_for_epoch
from utils import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_stack(name: str, overlays: tuple[str, ...] = ()) -> list[str]:
    selected = {Path(path).parent.name: path for path in overlays}
    prefix = f"configs/ablations/{name}"
    future, known = (180, 60) if name == "issm" else (64, 8)
    return [
        "configs/default.yaml",
        f"configs/datasets/{name}.yaml",
        f"configs/protocols/{name}/main.yaml",
        "configs/models/node2.yaml",
        selected.get("history", f"{prefix}/history/h1.yaml"),
        selected.get("architecture", f"{prefix}/architecture/full.yaml"),
        selected.get("training_horizon", f"{prefix}/training_horizon/k{future}.yaml"),
        selected.get("rollout_start", f"{prefix}/rollout_start/known{known}.yaml"),
        selected.get("temporal_consistency", f"{prefix}/temporal_consistency/tc0.yaml"),
        "configs/runtime/fast.yaml",
    ]


def _config(name: str, data_dir: Path, overlays: tuple[str, ...] = ()) -> dict:
    config = load_config_bundle([PROJECT_ROOT / path for path in _config_stack(name, overlays)])
    config["dataset"]["data_dir"] = str(data_dir)
    return config


def validate_temporal_settings(config: dict, lengths: dict[str, int]) -> dict:
    """Validate training horizon availability and rollout starts separately."""

    history = int(config["dataset"]["history_len"])
    future = int(config["dataset"]["future_len"])
    known = int(config["evaluation"]["known_steps"])
    training = config["training"]
    minimum = int(training.get("train_horizon_min", 1))
    maximum = training.get("train_horizon_max")
    if history < 1 or future < 1:
        raise ValueError("history_len and future_len must be >= 1.")
    if known < history:
        raise ValueError("evaluation.known_steps must be >= dataset.history_len.")
    if not lengths:
        raise ValueError("Temporal audit requires at least one trajectory.")
    for scenario, length in lengths.items():
        if known >= length:
            raise ValueError(f"evaluation.known_steps must be < trajectory length: {scenario} has {length} steps.")
    if minimum < 1 or future < minimum:
        raise ValueError("dataset.future_len must be >= training.train_horizon_min >= 1.")
    if maximum is not None and not minimum <= int(maximum) <= future:
        raise ValueError("training.train_horizon_max must be between train_horizon_min and dataset.future_len.")
    return {
        "history_len": history,
        "future_len": future,
        "known_steps": known,
        "history_indices": list(range(known - history, known)),
        "rollout_lengths": sorted({length - known for length in lengths.values()}),
        "train_horizon_min": minimum,
        "target_train_horizon_max": future if maximum is None else int(maximum),
        "relative_time_scale": config["model"]["relative_time_scale"],
        "passed": True,
    }


def validate_main_protocol(config: dict, name: str) -> None:
    expected = {"issm": (1, 180, 60, 180.0, 60), "anuga": (1, 64, 8, 65.0, 9)}[name]
    actual = (config["dataset"]["history_len"], config["dataset"]["future_len"],
              config["evaluation"]["known_steps"], config["model"]["relative_time_scale"],
              config["dataset"]["train_series_per_scenario_per_epoch"])
    if actual != expected:
        raise ValueError(f"{name} main protocol must resolve to H/K/known/scale/budget={expected}, got {actual}.")
    if config["dataset"]["name"] != name:
        raise ValueError(f"Expected dataset identity {name}.")
    if name == "issm" and Path(config["dataset"]["data_dir"]).name != "PIG_5000":
        raise ValueError("The standard ISSM dataset must be PIG_5000.")


def _splits(config: dict) -> dict[str, list[Path]]:
    return dict(zip(("train", "val", "test"), build_splits(config)))


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def _audit_issm(config: dict) -> tuple[dict, dict[str, int]]:
    validate_main_protocol(config, "issm")
    known_steps = int(config["evaluation"]["known_steps"])
    splits = _splits(config)
    files = sorted(path for paths in splits.values() for path in paths)
    membership = {path: split for split, paths in splits.items() for path in paths}
    metadata = []
    leakage = []
    lengths = {}
    for index, path in enumerate(files):
        payload = _load_container(path)
        fields = payload["S"][0][0]
        thickness = np.asarray(fields[9], dtype=np.float32)
        lengths[path.name] = len(thickness)
        validate_temporal_settings(config, {path.name: len(thickness)})
        metadata.append({"file": path.name, "split": membership[path], "snapshots": int(thickness.shape[0]),
                         "nodes": int(thickness.shape[1]), "melt_rate": parse_issm_rate_from_filename(path)})
        if membership[path] in {"val", "test"}:
            floating = np.asarray(fields[10], dtype=np.float32)
            anchor = known_steps - 1
            reconstruction = thickness[anchor] + floating[anchor + 1:] - floating[anchor]
            error = reconstruction.astype(np.float64) - thickness[anchor + 1:].astype(np.float64)
            leakage.append({"file": path.name, "split": membership[path], "t_end": anchor,
                            "future_steps": int(error.shape[0]), "float32_reconstruction_rmse": float(np.sqrt(np.mean(error ** 2))),
                            "max_abs_error": float(np.max(np.abs(error)))})
            del floating, reconstruction, error
        del payload, fields, thickness
        print(f"ISSM metadata {index + 1}/{len(files)}: {path.name}", flush=True)

    sample_path = splits["train"][0]
    dataset = ISSMDataset([sample_path], history_len=config["dataset"]["history_len"],
                          future_len=config["dataset"]["future_len"], split="train", cache_in_memory=True,
                          adapter_kwargs=config["dataset"].get("issm", {}))
    trajectory = dataset.iter_trajectories()[0]
    sample = dataset.get_rollout_data(0, start_t=known_steps - 1)
    result = {
        "data_dir": config["dataset"]["data_dir"],
        "split_rates": {split: sorted({parse_issm_rate_from_filename(path) for path in paths}) for split, paths in splits.items()},
        "split_counts": {split: len(paths) for split, paths in splits.items()},
        "trajectory_metadata": metadata,
        "snapshot_count_histogram": dict(Counter(lengths.values())),
        "sample_file": sample_path.name,
        "channels": {"x_static": ["base0", "surface0", "speed0", "floating_ocean_mask0"],
                     "force": ["basal_melt", "SMB"], "state": ["vx", "vy", "thickness"]},
        "trajectory_shapes": {"x_static": list(trajectory.x_static.shape), "force": list(trajectory.force.shape),
                              "state": list(trajectory.state.shape)},
        "sample_shapes": {name: list(getattr(sample, name).shape)
                          for name in ["x_static", "state_hist", "force_hist", "force_future", "y_future"]},
        "static_channel_statistics": _stats(trajectory.x_static),
        "static_first_three_nodes": trajectory.x_static[:3].tolist(),
        "force_first_three_nodes_at_t0": trajectory.force[0, :3].tolist(),
        "mask_values": np.unique(trajectory.x_static[:, 3]).tolist(),
        "mask_convention": "1 = floating/ocean, defined by raw floating[0] < 0; 0 otherwise",
        "future_floating_available_to_model": False,
        "coordinates_predictive_channels": False,
        "edge_attr_consumed_by_node2": False,
        "raw_analytic_leakage_formula": f"H[{known_steps - 1}] + floating[{known_steps}:] - floating[{known_steps - 1}] (arithmetic in float32)",
        "raw_analytic_leakage": leakage,
        "temporal_semantics": validate_temporal_settings(config, lengths),
    }
    return result, lengths


def _audit_anuga(config: dict) -> tuple[dict, dict[str, int]]:
    validate_main_protocol(config, "anuga")
    splits = _splits(config)
    membership = {path: split for split, paths in splits.items() for path in paths}
    metadata = []
    lengths = {}
    rainfall = []
    node_counts = []
    for path in sorted(membership):
        with np.load(path) as payload:
            times = payload["time"]
            nodes = len(payload["x"])
            lengths[path.name] = len(times)
            metadata.append({"file": path.name, "split": membership[path], "snapshots": len(times), "nodes": nodes})
            if membership[path] == "train":
                rainfall.append(np.asarray(payload["rain_rate"], dtype=np.float32))
                node_counts.append(nodes)
    # Each node receives identical rainfall. Divide all multiplicities by their
    # common divisor to preserve exactly the full trajectory population weights
    # while avoiding loading any water-state fields or repeated mesh rainfall.
    divisor = math.gcd(*node_counts)
    trajectories = []
    for rain, nodes in zip(rainfall, node_counts):
        compact_force = np.repeat(rain[:, None], nodes // divisor, axis=1)[..., None]
        trajectories.append(SimpleNamespace(x_static=np.zeros((1, 1), dtype=np.float32),
                                             state=np.zeros((1, 1, 1), dtype=np.float32), force=compact_force))
    normalizer = FeatureNormalizer.fit_from_trajectories(trajectories)
    raw = _RunningStats(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    normalized = _RunningStats(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    for trajectory in trajectories:
        raw.update(trajectory.force)
        normalized.update(normalizer.transform_force(torch.from_numpy(trajectory.force)).numpy())
    raw_std = torch.sqrt(raw.m2 / raw.count)
    normalized_std = torch.sqrt(normalized.m2 / normalized.count)
    result = {
        "data_dir": config["dataset"]["data_dir"],
        "split_files": {split: [path.name for path in paths] for split, paths in splits.items()},
        "split_counts": {split: len(paths) for split, paths in splits.items()},
        "trajectory_metadata": metadata,
        "snapshot_count_histogram": dict(Counter(lengths.values())),
        "rainfall_units": "m/s (ANUGA Rate_operator SI rate, no adapter conversion)",
        "statistics_scope": "All training-split times and nodes, weighted exactly as full adapter force tensors",
        "compact_weight_common_divisor": divisor,
        "full_training_force_value_count": sum(len(rain) * nodes for rain, nodes in zip(rainfall, node_counts)),
        "raw_training_rainfall": {"mean": raw.mean.item(), "std": raw_std.item(),
                                  "min": min(float(rain.min()) for rain in rainfall),
                                  "max": max(float(rain.max()) for rain in rainfall)},
        "normalizer_force_mean": normalizer.force_mean.tolist(),
        "normalizer_force_std": normalizer.force_std.tolist(),
        "normalized_training_rainfall": {"mean": normalized.mean.item(), "std": normalized_std.item()},
        "normalizer_scope": "Force statistics fitted with FeatureNormalizer; dummy static/state arrays unused",
        "temporal_semantics": validate_temporal_settings(config, lengths),
    }
    print(f"ANUGA train rainfall: raw mean/std={raw.mean.item():.9g}/{raw_std.item():.9g}; "
          f"normalized mean/std={normalized.mean.item():.9g}/{normalized_std.item():.9g}", flush=True)
    return result, lengths


def _scan_table(name: str, axis: str, data_dir: Path, lengths: dict[str, int]) -> dict:
    rows = []
    for path in sorted((PROJECT_ROOT / "configs/ablations" / name / axis).glob("*.yaml")):
        overlays = (str(path.relative_to(PROJECT_ROOT)),)
        config = _config(name, data_dir, overlays)
        dataset_cfg = config["dataset"]
        semantics = validate_temporal_settings(config, lengths)
        train_files = _splits(config)["train"]
        budget = int(dataset_cfg["train_series_per_scenario_per_epoch"])
        per_scenario = {}
        natural_series = []
        for scenario_index, scenario_path in enumerate(train_files):
            anchors = enumerate_window_end_indices(
                lengths[scenario_path.name], dataset_cfg["history_len"], dataset_cfg["future_len"],
                dataset_cfg.get("stride", 1),
            )
            if not anchors:
                raise ValueError(f"No natural training anchors for {scenario_path.name}: {path}")
            per_scenario[scenario_path.name] = anchors
            natural_series.extend(WindowMetadata(scenario_index, anchor) for anchor in anchors)
        selected = sample_training_series_for_epoch(
            natural_series, len(train_files), np.random.default_rng(config["seed"]), budget,
        )
        selected_counts = Counter(series.scenario_index for series in selected)
        assert set(selected_counts.values()) == {budget}
        total_series = len(selected)
        assert total_series == len(train_files) * budget
        assert total_series % 4 == 0
        natural_counts = sorted({len(anchors) for anchors in per_scenario.values()})
        first_anchors = sorted({anchors[0] for anchors in per_scenario.values()})
        last_anchors = sorted({anchors[-1] for anchors in per_scenario.values()})
        batch_size = int(config["training"]["batch_size"])
        accumulation = int(config["training"].get("grad_accum_steps", 1))
        steps = lambda world_size: math.ceil(math.ceil(math.ceil(total_series / world_size) / batch_size) / accumulation)
        rows.append({"config": str(path.relative_to(PROJECT_ROOT)), "history_len": dataset_cfg["history_len"],
                     "future_len": dataset_cfg["future_len"], "known_steps": semantics["known_steps"],
                     "rollout_lengths": semantics["rollout_lengths"], "natural_anchor_counts": natural_counts,
                     "first_t_end": first_anchors, "last_t_end": last_anchors,
                     "training_scenarios": len(train_files), "total_natural_anchors": len(natural_series),
                     "train_series_per_scenario_per_epoch": budget,
                     "total_training_series_per_epoch": total_series,
                     "four_gpu_sampler_padding": 0,
                     "per_rank_batch_size": batch_size, "grad_accum_steps": accumulation,
                     "optimizer_steps_per_epoch_single_process": steps(1),
                     "optimizer_steps_per_epoch_world_sizes": {str(world): steps(world) for world in [1, 2, 4, 8]}})
    if not rows:
        raise ValueError(f"No configs found for {name}/{axis}")
    rows.sort(key=lambda row: (row["history_len"], row["future_len"]))
    assert len({row["total_training_series_per_epoch"] for row in rows}) == 1
    return {"training_anchor_rule": "range(history_len - 1, trajectory_length - future_len, stride)",
            "sampling_rule": "N >= B: B distinct anchors; N < B: all N once plus B-N replacement draws, shuffled per scenario",
            "config_stack_template": _config_stack(name, (f"configs/ablations/{name}/{axis}/<variant>.yaml",)),
            "history_note": "K tables use H1 for this structural audit; formal Phase 3 requires the selected Phase-1 H*.",
            "data_dir": str(data_dir),
            "sbatch_default_world_size": 4,
            "optimizer_steps_formula": "ceil(ceil(ceil(total_training_series_per_epoch / world_size) / per_rank_batch_size) / grad_accum_steps)",
            "training_sampler_note": "Formal totals divide by four without padding; other world sizes use PyTorch DistributedSampler semantics.",
            "variants": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ISSM/ and ANUGA/.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/foundation_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    issm_dir = args.data_root / "ISSM/PIG_5000"
    anuga_dir = args.data_root / "ANUGA/simulation_data_merged"
    issm, issm_lengths = _audit_issm(_config("issm", issm_dir))
    anuga, anuga_lengths = _audit_anuga(_config("anuga", anuga_dir))
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"python scripts/audit_foundation.py --data-root {args.data_root} --output {args.output}",
        "scope": "Read-only real-data audit; CPU; no optimization/training or accuracy claim",
        "formal_config_sources": {name: _config_stack(name) for name in ("issm", "anuga")},
        "dataset_relocation_note": "Preserve formal subdirectory choices while relocating their data root via --data-root.",
        "issm": issm,
        "anuga": anuga,
        "scans": {
            "anuga_history": _scan_table("anuga", "history", anuga_dir, anuga_lengths),
            "issm_history": _scan_table("issm", "history", issm_dir, issm_lengths),
            "issm_training_horizon": _scan_table("issm", "training_horizon", issm_dir, issm_lengths),
            "anuga_training_horizon": _scan_table("anuga", "training_horizon", anuga_dir, anuga_lengths),
        },
        "relative_time_prefix": "Separate model regression/audit; this script does not run model inference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
