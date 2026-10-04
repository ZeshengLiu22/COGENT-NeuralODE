#!/usr/bin/env python3
"""Fixed-window evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch
from torch_geometric.loader import DataLoader

from datasets.factory import build_dataset, build_splits
from datasets.normalization import FeatureNormalizer
from models import build_model
from utils import configure_logging, load_config_bundle
from utils.eval_artifacts import (
    collect_window_prediction_bundle,
    format_metric_table,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_window_bundle,
)


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
        "--future-len",
        type=int,
        default=None,
        help="Optional evaluation-only future length override. Defaults to dataset.future_len from the config.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for window evaluation, but no CUDA device is available.")
    return device


def main() -> None:
    args = parse_args()
    config = load_config_bundle(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    logger = configure_logging()
    device = resolve_device(args.device)

    dataset_name = config["dataset"]["name"]
    train_history_len = int(config["dataset"]["history_len"])
    train_future_len = int(config["dataset"]["future_len"])
    eval_history_len = train_history_len if args.history_len is None else int(args.history_len)
    eval_future_len = train_future_len if args.future_len is None else int(args.future_len)
    if eval_history_len < 1:
        raise ValueError(f"--history-len must be >= 1, got {eval_history_len}.")
    if eval_future_len < 1:
        raise ValueError(f"--future-len must be >= 1, got {eval_future_len}.")
    train_files, val_files, test_files = build_splits(config)
    split_files = {"train": train_files, "val": val_files, "test": test_files}
    dataset = build_dataset(
        dataset_name,
        split_files[args.split],
        args.split,
        config,
        normalizer,
        history_len=eval_history_len,
        future_len=eval_future_len,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No evaluation windows are available for split={args.split} with "
            f"history_len={eval_history_len} and future_len={eval_future_len}."
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["evaluation"].get("num_workers", 0)),
    )

    sample = dataset[0]
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
        "Window evaluation lengths: history_len=%d future_len=%d (training config: history_len=%d future_len=%d)",
        eval_history_len,
        eval_future_len,
        train_history_len,
        train_future_len,
    )
    bundle = collect_window_prediction_bundle(model, loader, normalizer, device=device)
    channel_names = infer_state_channel_names(dataset_name, state_dim=int(bundle["pred_phys"].shape[1]))
    summary = summarize_window_bundle(bundle, channel_names)
    artifact_stem = Path(str(Path(args.checkpoint).with_suffix("")) + ".window")
    artifact_paths = save_evaluation_artifacts(
        artifact_stem,
        bundle=bundle,
        summary=summary,
        dataset_name=dataset_name,
        split=args.split,
        mode="fixed_window",
        channel_names=channel_names,
        scenario_infos=dataset.scenario_infos,
        metadata={
            "eval_history_len": eval_history_len,
            "eval_future_len": eval_future_len,
            "train_config_history_len": train_history_len,
            "train_config_future_len": train_future_len,
        },
    )

    logger.info("Evaluation metrics: %s", summary["metrics"])
    logger.info("Detailed metric table:\n%s", format_metric_table(summary["metric_table"]))
    logger.info("Saved evaluation artifacts: %s", artifact_paths)


if __name__ == "__main__":
    main()
