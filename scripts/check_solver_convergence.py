#!/usr/bin/env python3
"""Post-training midpoint refinement check on the checkpoint's validation split."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch

from datasets.factory import build_dataset
from datasets.normalization import FeatureNormalizer
from models import build_model
from utils import configure_logging, save_json
from utils.checkpoint_evaluation import resolve_device, restore_checkpoint_splits, restore_evaluation_config


@torch.no_grad()
def check_convergence(model, dataset, normalizer, device: torch.device, *, start_t: int) -> dict:
    """Compare the saved midpoint grid, half-step, and quarter-step solutions."""

    original_solver = deepcopy(model.solver_cfg)
    if original_solver.get("ode_method", "midpoint") != "midpoint":
        raise ValueError("This diagnostic requires a checkpoint trained with the midpoint solver.")
    if not dataset.scenario_infos:
        raise ValueError("The checkpoint validation split is empty.")
    if start_t < dataset.history_len - 1:
        raise ValueError("The rollout start must include the configured observed history.")
    settings = [("baseline", None), ("step_size_0.5", 0.5), ("step_size_0.25", 0.25)]
    rows = [{
        "name": name,
        "ode_options": deepcopy(original_solver.get("ode_options")) if step is None else {
            **(deepcopy(original_solver.get("ode_options")) or {}), "step_size": step,
        },
        "runtime_seconds": 0.0,
        "physical_sq_error": 0.0,
        "normalized_sq_error": 0.0,
        "physical_sq_difference": 0.0,
        "normalized_sq_difference": 0.0,
        "max_absolute_prediction_difference": 0.0,
        "max_absolute_normalized_prediction_difference": 0.0,
    } for name, step in settings]
    count = 0
    model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for index in range(len(dataset.scenario_infos)):
            sample = dataset.get_rollout_data(index, start_t=start_t).to(device)
            target = sample.y_future.float()
            target_phys = normalizer.inverse_state(target).double()
            predictions = []
            for row in rows:
                model.solver_cfg = {**original_solver, "ode_options": deepcopy(row["ode_options"])}
                synchronize()
                started = perf_counter()
                prediction = model(sample).float()
                synchronize()
                row["runtime_seconds"] += perf_counter() - started
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite predictions for {row['name']}.")
                physical = normalizer.inverse_state(prediction).double()
                row["physical_sq_error"] += (physical - target_phys).square().sum().item()
                row["normalized_sq_error"] += (prediction.double() - target.double()).square().sum().item()
                predictions.append((prediction.double(), physical))
            # Each coarser setting is compared with the next finer solution.
            for setting_index, row in enumerate(rows):
                reference_index = min(setting_index + 1, len(rows) - 1)
                diff_norm = predictions[setting_index][0] - predictions[reference_index][0]
                diff_phys = predictions[setting_index][1] - predictions[reference_index][1]
                row["physical_sq_difference"] += diff_phys.square().sum().item()
                row["normalized_sq_difference"] += diff_norm.square().sum().item()
                row["max_absolute_prediction_difference"] = max(row["max_absolute_prediction_difference"], diff_phys.abs().max().item())
                row["max_absolute_normalized_prediction_difference"] = max(row["max_absolute_normalized_prediction_difference"], diff_norm.abs().max().item())
                row["reference_solution"] = rows[reference_index]["name"]
            count += target.numel()
    finally:
        model.solver_cfg = original_solver

    for row in rows:
        row["validation_physical_rmse"] = math.sqrt(row.pop("physical_sq_error") / count)
        row["validation_normalized_rmse"] = math.sqrt(row.pop("normalized_sq_error") / count)
        row["prediction_rmse_difference"] = math.sqrt(row.pop("physical_sq_difference") / count)
        row["normalized_prediction_rmse_difference"] = math.sqrt(row.pop("normalized_sq_difference") / count)
    fine_error = rows[-1]["validation_normalized_rmse"]
    fine_difference = rows[1]["normalized_prediction_rmse_difference"]
    return {
        "split": "val",
        "mode": "full_rollout",
        "amp_mode": "none",
        "known_steps": start_t + 1,
        "scenario_count": len(dataset.scenario_infos),
        "scalar_prediction_count": count,
        "settings": rows,
        "half_to_quarter_difference_over_model_error": fine_difference / fine_error if fine_error > 0.0 else None,
        "interpretation": (
            "Compare the 0.5-to-0.25 normalized prediction difference with the quarter-step validation model error. "
            "A negligible ratio supports numerical convergence. Solver settings are never changed by this diagnostic. "
            "Physical aggregate errors mix state units; use normalized errors to assess relative adequacy. "
            "The baseline comparison uses the saved grid, which must be interpreted from its ode_options."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    overrides = {"dataset": {"data_dir": args.data_dir}} if args.data_dir is not None else None
    config = restore_evaluation_config(checkpoint, runtime_overrides=overrides)
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    split_files = restore_checkpoint_splits(checkpoint, config["dataset"]["data_dir"])
    dataset = build_dataset(config["dataset"]["name"], split_files["val"], "val", config, normalizer)
    if not dataset.scenario_infos:
        raise ValueError("The checkpoint validation split is empty.")
    known_steps = config["evaluation"].get("full_rollout_known_steps") or dataset.history_len
    sample = dataset.get_rollout_data(0, start_t=int(known_steps) - 1)
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(args.device)
    model.to(device)
    result = check_convergence(model, dataset, normalizer, device, start_t=int(known_steps) - 1)
    result["checkpoint"] = str(checkpoint_path.resolve())
    result["config_source"] = "checkpoint"
    result["split_source"] = "checkpoint.split_manifest"
    output = Path(args.output) if args.output else checkpoint_path.parent / "solver_convergence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    logger = configure_logging()
    logger.info("Saved validation midpoint convergence diagnostic: %s", output)
    logger.info("Half-to-quarter normalized change / model error: %s", result["half_to_quarter_difference_over_model_error"])


if __name__ == "__main__":
    main()
