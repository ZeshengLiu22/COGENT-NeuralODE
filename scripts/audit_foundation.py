#!/usr/bin/env python3
"""Read-only, CPU audit of repaired inputs and formal ablation window counts.

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
from datasets.window_utils import enumerate_window_end_indices
from utils import load_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(relative_path: str, data_dir: Path, overlays: tuple[str, ...] = ()) -> dict:
    config = load_config_bundle([PROJECT_ROOT / relative_path] + [PROJECT_ROOT / overlay for overlay in overlays])
    config["dataset"]["data_dir"] = str(data_dir)
    return config


def _splits(config: dict) -> dict[str, list[Path]]:
    return dict(zip(("train", "val", "test"), build_splits(config)))


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def _audit_issm(config: dict) -> tuple[dict, dict[str, int]]:
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
        metadata.append({"file": path.name, "split": membership[path], "snapshots": int(thickness.shape[0]),
                         "nodes": int(thickness.shape[1]), "melt_rate": parse_issm_rate_from_filename(path)})
        if membership[path] in {"val", "test"}:
            floating = np.asarray(fields[10], dtype=np.float32)
            anchor = 59
            reconstruction = thickness[anchor] + floating[anchor + 1:] - floating[anchor]
            error = reconstruction.astype(np.float64) - thickness[anchor + 1:].astype(np.float64)
            leakage.append({"file": path.name, "split": membership[path], "t_end": anchor,
                            "future_steps": int(error.shape[0]), "float32_reconstruction_rmse": float(np.sqrt(np.mean(error ** 2))),
                            "max_abs_error": float(np.max(np.abs(error)))})
            del floating, reconstruction, error
        del payload, fields, thickness
        print(f"ISSM metadata {index + 1}/{len(files)}: {path.name}", flush=True)

    sample_path = splits["train"][0]
    dataset = ISSMDataset([sample_path], history_len=6, future_len=8, split="train", cache_in_memory=True)
    trajectory = dataset.iter_trajectories()[0]
    sample = dataset.get_rollout_data(0, start_t=59)
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
        "raw_analytic_leakage_formula": "H[59] + floating[60:] - floating[59] (arithmetic in float32)",
        "raw_analytic_leakage": leakage,
        "full_rollout_known_steps": 60,
        "full_rollout_future_steps": sorted({length - 60 for length in lengths.values()}),
        "relative_time_scale": config["model"]["relative_time_scale"],
    }
    return result, lengths


def _audit_anuga(config: dict) -> tuple[dict, dict[str, int]]:
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
        "full_rollout_known_steps": 8,
        "full_rollout_future_steps": sorted({length - 8 for length in lengths.values()}),
        "relative_time_scale": config["model"]["relative_time_scale"],
    }
    print(f"ANUGA train rainfall: raw mean/std={raw.mean.item():.9g}/{raw_std.item():.9g}; "
          f"normalized mean/std={normalized.mean.item():.9g}/{normalized_std.item():.9g}", flush=True)
    return result, lengths


def _scan_table(directory: str, data_dir: Path, lengths: dict[str, int], overlays: tuple[str, ...] = ()) -> dict:
    rows = []
    expected_anchors = None
    for path in sorted((PROJECT_ROOT / "configs" / directory).glob("base_*.yaml")):
        config = _config(str(path.relative_to(PROJECT_ROOT)), data_dir, overlays)
        dataset_cfg = config["dataset"]
        train_files = _splits(config)["train"]
        per_scenario = {}
        for scenario_path in train_files:
            per_scenario[scenario_path.name] = enumerate_window_end_indices(
                lengths[scenario_path.name], dataset_cfg["history_len"], dataset_cfg["future_len"],
                dataset_cfg.get("stride", 1), dataset_cfg.get("window_reference"),
            )
        if expected_anchors is None:
            expected_anchors = per_scenario
        if expected_anchors != per_scenario:
            raise AssertionError(f"Prediction anchors differ within {directory}: {path.name}")
        window_counts = sorted({len(anchors) for anchors in per_scenario.values()})
        first_anchors = sorted({anchors[0] for anchors in per_scenario.values() if anchors})
        last_anchors = sorted({anchors[-1] for anchors in per_scenario.values() if anchors})
        total_windows = sum(len(anchors) for anchors in per_scenario.values())
        batch_size = int(config["training"]["batch_size"])
        accumulation = int(config["training"].get("grad_accum_steps", 1))
        steps = lambda world_size: math.ceil(math.ceil(math.ceil(total_windows / world_size) / batch_size) / accumulation)
        rows.append({"config": str(path.relative_to(PROJECT_ROOT)), "history_len": dataset_cfg["history_len"],
                     "future_len": dataset_cfg["future_len"], "windows_per_scenario": window_counts,
                     "first_t_end": first_anchors, "last_t_end": last_anchors,
                     "training_scenarios": len(train_files), "total_training_windows": total_windows,
                     "per_rank_batch_size": batch_size, "grad_accum_steps": accumulation,
                     "optimizer_steps_per_epoch_single_process": steps(1),
                     "optimizer_steps_per_epoch_world_sizes": {str(world): steps(world) for world in [1, 2, 4, 8]}})
    rows.sort(key=lambda row: (row["history_len"], row["future_len"]))
    return {"identical_prediction_anchors": True,
            "config_overlays_in_order": list(overlays), "data_dir": str(data_dir),
            "sbatch_default_world_size": 4,
            "optimizer_steps_formula": "ceil(ceil(ceil(total_training_windows / world_size) / per_rank_batch_size) / grad_accum_steps)",
            "training_sampler_note": "Distributed training retains PyTorch padding; evaluation never pads.", "variants": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ISSM/ and ANUGA/.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/foundation_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    issm_dir = args.data_root / "ISSM/PIG_data"
    issm_formal_dir = args.data_root / "ISSM/PIG_5000"
    anuga_dir = args.data_root / "ANUGA/simulation_data_merged"
    issm_config_path = "configs/ISSM_History_Scan/base_ISSM_history_8.yaml"
    anuga_config_path = "configs/ANUGA_History_Scan/base_ANUGA_history8.yaml"
    issm_overlays = ("configs/model_node2.yaml", "configs/ISSM_History_Scan/issm_pig5000_fast_loader.yaml")
    anuga_overlays = ("configs/model_node2.yaml", "configs/ANUGA_History_Scan/anuga_fast_loader.yaml")
    issm, issm_lengths = _audit_issm(_config(issm_config_path, issm_dir))
    issm_formal, issm_formal_lengths = _audit_issm(_config(issm_config_path, issm_formal_dir, issm_overlays))
    anuga, anuga_lengths = _audit_anuga(_config(anuga_config_path, anuga_dir, anuga_overlays))
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"python scripts/audit_foundation.py --data-root {args.data_root} --output {args.output}",
        "scope": "Read-only real-data audit; CPU; no optimization/training or accuracy claim",
        "formal_config_sources": {"issm": [issm_config_path, *issm_overlays],
                                  "anuga": [anuga_config_path, *anuga_overlays]},
        "dataset_relocation_note": "Preserve formal subdirectory choices while relocating their data root via --data-root.",
        "issm": issm,
        "issm_formal_pig5000": issm_formal,
        "anuga": anuga,
        "scans": {
            "anuga_history": _scan_table("ANUGA_History_Scan", anuga_dir, anuga_lengths, anuga_overlays),
            "issm_history": _scan_table("ISSM_History_Scan", issm_formal_dir, issm_formal_lengths, issm_overlays),
            "issm_future": _scan_table("ISSM_Future_Len_Ablation", issm_formal_dir, issm_formal_lengths, issm_overlays),
        },
        "relative_time_prefix": "Separate model regression/audit; this script does not run model inference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
