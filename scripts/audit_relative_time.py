"""Read-only real-trajectory prefix check with a compact, untrained NODE2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.factory import build_dataset, build_splits
from datasets.normalization import FeatureNormalizer
from models import build_model
from training.trainer import _truncate_future_horizon
from utils.io import load_config_bundle, save_json


def audit_dataset(name: str, data_root: Path, seed: int) -> dict:
    config = load_config_bundle([
        PROJECT_ROOT / "configs/default.yaml",
        PROJECT_ROOT / f"configs/datasets/{name}.yaml",
        PROJECT_ROOT / f"configs/protocols/{name}/main.yaml",
        PROJECT_ROOT / "configs/models/node2.yaml",
    ])
    subdirectory = "ISSM/PIG_5000" if name == "issm" else "ANUGA/simulation_data_merged"
    config["dataset"].update({
        "data_dir": str(data_root / subdirectory), "cache_in_memory": True,
    })
    # Only this numerical audit uses small widths; formal configs are unchanged.
    config["model"].update({"latent_dim": 16, "decoder_hidden_dims": [16]})
    config["model"]["continuous"]["hidden_dim"] = 16
    config["model"]["history_encoder"].update({
        "static_hidden_dims": [16], "static_embed_dim": 16, "gnn_hidden_dim": 16,
        "lstm_hidden_dim": 16, "history_transformer_ff_dim": 64,
    })
    train, _, _ = build_splits(config)
    if not train:
        raise ValueError(f"No training trajectories found for {name}")
    dataset = build_dataset(name, train[:1], "train", config, build_training_series=False)
    trajectory = next(iter(dataset.iter_trajectories()))
    normalizer = FeatureNormalizer.fit_from_trajectories([trajectory])
    dataset.normalizer = normalizer
    assert not dataset.all_windows and not dataset.active_windows
    full_rollout = dataset.get_rollout_data(0, start_t=int(config["evaluation"]["known_steps"]) - 1)
    expected_steps = int(trajectory.state.shape[0]) - int(config["evaluation"]["known_steps"])
    assert full_rollout.y_future.shape[1] == expected_steps
    sample = _truncate_future_horizon(full_rollout.clone(), 8)
    shorter = _truncate_future_horizon(full_rollout.clone(), 4)
    torch.manual_seed(seed)
    model = build_model(
        config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1],
    ).eval()
    with torch.no_grad():
        prediction4 = model(shorter)
        prediction8 = model(sample)
        normalized_difference = prediction4 - prediction8[:, :4]
        physical_difference = normalizer.inverse_state(prediction4) - normalizer.inverse_state(prediction8[:, :4])
    torch.testing.assert_close(prediction4, prediction8[:, :4], rtol=1e-6, atol=1e-6)
    return {
        "dataset": name,
        "trajectory": str(train[0]),
        "normalization": "Fitted on this single training trajectory for the audit only",
        "untrained_compact_model": True,
        "device": "cpu",
        "seed": seed,
        "model_config": config["model"],
        "solver_config": config["solver"],
        "total_snapshots": int(trajectory.state.shape[0]),
        "nodes": int(sample.x_static.shape[0]),
        "history_len": int(sample.state_hist.shape[1]),
        "requested_future_lengths": [4, 8],
        "rollout_to_end_steps": expected_steps,
        "training_future_len": config["dataset"]["future_len"],
        "training_anchors_constructed": len(dataset.all_windows),
        "relative_time_scale": model.dynamics.relative_time_scale,
        "static_channels": int(sample.x_static.shape[-1]),
        "forcing_channels": int(sample.force_hist.shape[-1]),
        "state_channels": int(sample.state_hist.shape[-1]),
        "normalized_max_abs_prediction_difference": float(normalized_difference.abs().max()),
        "normalized_prediction_rmse_difference": float(normalized_difference.square().mean().sqrt()),
        "physical_max_abs_prediction_difference": float(physical_difference.abs().max()),
        "physical_prediction_rmse_difference": float(physical_difference.square().mean().sqrt()),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("issm", "anuga", "both"), default="both")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("docs/repair_validation/real_prefix.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    names = ("issm", "anuga") if args.dataset == "both" else (args.dataset,)
    results = {name: audit_dataset(name, args.data_root, args.seed) for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
