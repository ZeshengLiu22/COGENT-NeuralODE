#!/usr/bin/env python3
"""Full-trajectory rollout evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch

from datasets import ADCIRCDataset, ANUGADataset, ISSMDataset
from datasets.normalization import FeatureNormalizer
from datasets.split_utils import discover_files, issm_rate_modulo_split, random_split
from models import build_model
from utils import configure_logging, load_config_bundle
from utils.eval_artifacts import (
    collect_full_rollout_prediction_bundle,
    format_metric_table,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_full_rollout_bundle,
)


DATASET_REGISTRY = {
    "anuga": ANUGADataset,
    "adcirc": ADCIRCDataset,
    "issm": ISSMDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--history-len",
        type=int,
        default=None,
        help="Optional evaluation-only history length override. Defaults to dataset.history_len from the config.",
    )
    parser.add_argument(
        "--known-steps",
        type=int,
        default=None,
        help="Number of initial true timesteps before rollout begins. Defaults to the evaluation history length.",
    )
    parser.add_argument(
        "--plot-max-lead-step",
        type=int,
        default=None,
        help="Only save the first this many lead-time steps in the full-rollout summary and curve artifacts. The rollout itself still runs to the end.",
    )
    return parser.parse_args()


def build_split_files(config: dict) -> tuple[list[Path], list[Path], list[Path]]:
    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]
    files = discover_files(dataset_cfg["data_dir"], dataset_cfg["file_patterns"])
    strategy = str(split_cfg.get("strategy", "random")).lower()
    if strategy == "issm_rate_modulo":
        return issm_rate_modulo_split(files, modulo=int(split_cfg.get("modulo", 20)), val_remainder=int(split_cfg.get("val_remainder", 0)), test_remainder=int(split_cfg.get("test_remainder", 10)))
    return random_split(files, train=float(split_cfg["train"]), val=float(split_cfg["val"]), test=float(split_cfg["test"]), seed=int(split_cfg.get("seed", config["seed"])))


def build_dataset(
    dataset_name: str,
    files: list[Path],
    split: str,
    config: dict,
    normalizer,
    *,
    history_len: int | None = None,
):
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_cfg = config["dataset"]
    return dataset_cls(
        scenario_files=files,
        history_len=int(dataset_cfg["history_len"] if history_len is None else history_len),
        future_len=int(dataset_cfg["future_len"]),
        split=split,
        stride=int(dataset_cfg.get("stride", 1)),
        normalizer=normalizer,
        cache_in_memory=bool(dataset_cfg.get("cache_in_memory", False)),
        seed=int(dataset_cfg.get("seed", config["seed"])),
        adapter_kwargs=dataset_cfg.get(dataset_name, {}),
    )


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for full-rollout evaluation, but no CUDA device is available.")
    return device


def main() -> None:
    args = parse_args()
    config = load_config_bundle(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    logger = configure_logging()
    device = resolve_device(args.device)

    dataset_name = str(config["dataset"]["name"]).lower()
    train_history_len = int(config["dataset"]["history_len"])
    eval_history_len = train_history_len if args.history_len is None else int(args.history_len)
    if eval_history_len < 1:
        raise ValueError(f"--history-len must be >= 1, got {eval_history_len}.")
    known_steps = eval_history_len if args.known_steps is None else int(args.known_steps)
    if known_steps < eval_history_len:
        raise ValueError(
            f"--known-steps must be >= evaluation history length ({eval_history_len}), got {known_steps}."
        )
    train_files, val_files, test_files = build_split_files(config)
    split_files = {"train": train_files, "val": val_files, "test": test_files}
    dataset = build_dataset(
        dataset_name,
        split_files[args.split],
        args.split,
        config,
        normalizer,
        history_len=eval_history_len,
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

    sample = dataset.get_rollout_data(0, start_t=start_t)
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
    load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("Using evaluation device: %s (%s)", device, torch.cuda.get_device_name(device_index))
    else:
        logger.info("Using evaluation device: %s", device)
    logger.info(
        "Full-rollout evaluation context: history_len=%d known_steps=%d plot_max_lead_step=%s (training config history_len=%d)",
        eval_history_len,
        known_steps,
        "all" if args.plot_max_lead_step is None else str(int(args.plot_max_lead_step)),
        train_history_len,
    )
    if load_result.missing_keys:
        logger.warning("Missing state_dict keys during rollout load: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.warning("Unexpected state_dict keys during rollout load: %s", load_result.unexpected_keys)
    bundle = collect_full_rollout_prediction_bundle(model, dataset, normalizer, device=device, start_t=start_t)
    channel_names = infer_state_channel_names(dataset_name, state_dim=int(bundle["pred_phys"].shape[1]))
    summary_method = f"{config['model']['name']}_full_rollout"
    summary = summarize_full_rollout_bundle(
        bundle,
        channel_names,
        max_lead_time_step=args.plot_max_lead_step,
        summary_method=summary_method,
    )
    artifact_stem = Path(str(Path(args.checkpoint).with_suffix("")) + ".full_rollout")
    artifact_paths = save_evaluation_artifacts(
        artifact_stem,
        bundle=bundle,
        summary=summary,
        dataset_name=dataset_name,
        split=args.split,
        mode="full_rollout",
        channel_names=channel_names,
        scenario_infos=dataset.scenario_infos,
        metadata={
            "eval_history_len": eval_history_len,
            "known_steps": known_steps,
            "train_config_history_len": train_history_len,
            "plot_max_lead_step": args.plot_max_lead_step,
            "full_rollout_num_lead_steps": len(summary["metrics"]["horizon_rmse_curve"]),
            "artifact_num_lead_steps": len(summary["leadtime_metrics"]),
            "summary_method": summary_method,
        },
    )

    logger.info("Full-rollout metrics: %s", summary["metrics"])
    logger.info("Detailed metric table:\n%s", format_metric_table(summary["metric_table"]))
    logger.info("Saved evaluation artifacts: %s", artifact_paths)


if __name__ == "__main__":
    main()
