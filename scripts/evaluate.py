#!/usr/bin/env python3
"""Full-trajectory rollout evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

import torch

from datasets.factory import build_dataset
from datasets.normalization import FeatureNormalizer
from models import build_model
from utils import configure_logging, load_config_bundle
from utils.checkpoint_evaluation import evaluation_autocast, resolve_device, restore_checkpoint_splits, restore_evaluation_config
from utils.eval_artifacts import (
    collect_full_rollout_prediction_bundle,
    format_metric_table,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_full_rollout_bundle,
    validate_full_rollout_bundle,
)
from utils.anuga_postprocess import generate_anuga_flood_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", action="append", default=[], help="Runtime overrides, merged left to right; checkpoint science remains authoritative.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None, help="Device override; defaults to auto.")
    parser.add_argument("--data-dir", type=str, default=None, help="Relocate saved scenario identifiers to this directory.")
    parser.add_argument("--amp-mode", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--known-steps",
        type=int,
        default=None,
        help=(
            "Number of initial true timesteps before rollout begins. Defaults to "
            "evaluation.known_steps from the saved checkpoint config."
        ),
    )
    parser.add_argument(
        "--plot-max-lead-step",
        type=int,
        default=None,
        help="Only save the first this many lead-time steps in the full-rollout summary and curve artifacts. The rollout itself still runs to the end.",
    )
    parser.add_argument(
        "--anuga-output-dir",
        type=str,
        default=None,
        help="Optional directory for structured per-scenario ANUGA time series and flood maps.",
    )
    parser.add_argument(
        "--anuga-num-frames",
        type=int,
        default=3,
        help=(
            "Number of representative GT-vs-Pred flood-map frames per ANUGA scenario. "
            "Use 0 to save only the complete per-step time series."
        ),
    )
    parser.add_argument(
        "--skip-anuga-export",
        action="store_true",
        help="Skip automatic structured ANUGA time-series/flood-map export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    overrides = {"dataset": {}, "evaluation": {}}
    if args.data_dir is not None:
        overrides["dataset"]["data_dir"] = args.data_dir
    if args.amp_mode is not None:
        overrides["evaluation"]["amp_mode"] = args.amp_mode
    if args.known_steps is not None:
        overrides["evaluation"]["known_steps"] = args.known_steps
    external_config = load_config_bundle(args.config)
    config = restore_evaluation_config(checkpoint, external_config, runtime_overrides=overrides)
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    logger = configure_logging()
    device = resolve_device(args.device or config.get("device", "auto"))

    dataset_name = config["dataset"]["name"]
    train_history_len = int(config["dataset"]["history_len"])
    eval_history_len = train_history_len
    known_steps = int(config["evaluation"]["known_steps"])
    split_files = restore_checkpoint_splits(checkpoint, config["dataset"]["data_dir"])
    split_manifest = "checkpoint.split_manifest"
    missing_files = [path for path in split_files[args.split] if not path.is_file()]
    if missing_files:
        preview = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} saved {args.split} scenario file(s) are missing. First entries: {preview}"
        )
    dataset = build_dataset(
        dataset_name,
        split_files[args.split],
        args.split,
        config,
        normalizer,
        sample_windows=False,
    )
    if not dataset.scenario_infos:
        raise RuntimeError(f"No scenarios are available for split={args.split}.")
    min_steps = min(int(info["length"]) for info in dataset.scenario_infos)
    if known_steps >= min_steps:
        raise ValueError(
            f"--known-steps must leave at least one future step across the {args.split} split; "
            f"got known_steps={known_steps}, but the shortest trajectory has only {min_steps} steps."
        )
    start_t = known_steps - 1
    rollout_lengths = [int(info["length"]) - known_steps for info in dataset.scenario_infos]

    sample = dataset.get_rollout_data(0, start_t=start_t)
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("Using evaluation device: %s (%s)", device, torch.cuda.get_device_name(device_index))
    else:
        logger.info("Using evaluation device: %s", device)
    logger.info(
        "Full-rollout evaluation context: history_len=%d known_steps=%d rollout_steps=%d..%d "
        "plot_max_lead_step=%s (training config: history_len=%d future_len=%d)",
        eval_history_len,
        known_steps,
        min(rollout_lengths),
        max(rollout_lengths),
        "all" if args.plot_max_lead_step is None else str(int(args.plot_max_lead_step)),
        train_history_len,
        int(config["dataset"]["future_len"]),
    )
    logger.info("Evaluation provenance: checkpoint=%s split_manifest=%s", checkpoint_path, split_manifest)
    with evaluation_autocast(device, config["evaluation"]["amp_mode"]):
        bundle = collect_full_rollout_prediction_bundle(model, dataset, normalizer, device=device, known_steps=known_steps)
    coverage = validate_full_rollout_bundle(
        bundle,
        scenario_infos=dataset.scenario_infos,
        known_steps=known_steps,
        node_counts=[
            int(dataset.get_rollout_data(index, start_t=start_t).num_nodes)
            for index in range(len(dataset.scenario_infos))
        ],
    )
    channel_names = infer_state_channel_names(dataset_name, state_dim=int(bundle["pred_phys"].shape[1]))
    summary_method = "node2_full_rollout"
    summary = summarize_full_rollout_bundle(
        bundle,
        channel_names,
        max_lead_time_step=args.plot_max_lead_step,
        summary_method=summary_method,
    )
    artifact_stem = checkpoint_path.parent / f"{checkpoint_path.stem}.{args.split}.known{known_steps}.full_rollout"
    output_dir = args.output_dir or external_config.get("output_dir")
    if output_dir is not None:
        artifact_stem = Path(output_dir) / artifact_stem.name
    artifact_stem.parent.mkdir(parents=True, exist_ok=True)
    evaluation_metadata = {
        "history_len": eval_history_len,
        "eval_history_len": eval_history_len,
        "known_steps": known_steps,
        "rollout_start_idx0": start_t,
        "rollout_start_idx1": start_t + 1,
        "train_config_history_len": train_history_len,
        "train_config_future_len": int(config["dataset"]["future_len"]),
        "plot_max_lead_step": args.plot_max_lead_step,
        "full_rollout_num_lead_steps": len(summary["metrics"]["horizon_rmse_curve"]),
        "artifact_num_lead_steps": len(summary["leadtime_metrics"]),
        "summary_method": summary_method,
        "config_source": "checkpoint",
        "checkpoint": str(checkpoint_path),
        "runtime_config_stack": args.config,
        "relative_time_scale": config["model"].get("relative_time_scale"),
        "split_manifest": str(split_manifest),
        "amp_mode": config["evaluation"]["amp_mode"],
        "rollout_coverage": coverage,
    }
    artifact_paths = save_evaluation_artifacts(
        artifact_stem,
        bundle=bundle,
        summary=summary,
        dataset_name=dataset_name,
        split=args.split,
        mode="full_rollout",
        channel_names=channel_names,
        scenario_infos=dataset.scenario_infos,
        metadata=evaluation_metadata,
    )

    logger.info("Full-rollout metrics: %s", summary["metrics"])
    logger.info("Detailed metric table:\n%s", format_metric_table(summary["metric_table"]))
    logger.info("Saved evaluation artifacts: %s", artifact_paths)
    del bundle
    if dataset_name == "anuga" and not args.skip_anuga_export:
        flood_summary = generate_anuga_flood_maps(
            artifact_paths["predictions_npz"],
            predictions_meta_path=artifact_paths["predictions_meta_json"],
            output_dir=args.anuga_output_dir,
            num_frames=max(int(args.anuga_num_frames), 0),
            project_root=PROJECT_ROOT,
        )
        logger.info(
            "Saved complete per-step ANUGA GT/prediction time series and flood maps: %s",
            flood_summary["summary_json"],
        )


if __name__ == "__main__":
    main()
