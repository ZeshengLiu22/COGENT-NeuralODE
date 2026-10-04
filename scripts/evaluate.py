#!/usr/bin/env python3
"""Fixed-window evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch
from torch_geometric.loader import DataLoader

from datasets.factory import build_dataset
from datasets.normalization import FeatureNormalizer
from models import build_model
from utils import configure_logging, load_config_bundle
from utils.checkpoint_evaluation import evaluation_autocast, resolve_device, restore_checkpoint_splits, restore_evaluation_config
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
    parser.add_argument("--config", action="append", help="Optional runtime overrides. Model/config changes are rejected.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Relocate the saved scenario identifiers to this directory.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp-mode", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    overrides = {
        "dataset": {key: value for key, value in {
            "data_dir": args.data_dir, "history_len": args.history_len, "future_len": args.future_len,
        }.items() if value is not None},
        "evaluation": {key: value for key, value in {
            "batch_size": args.batch_size, "num_workers": args.num_workers, "amp_mode": args.amp_mode,
        }.items() if value is not None},
    }
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    external_config = load_config_bundle(args.config) if args.config else None
    config = restore_evaluation_config(checkpoint, external_config, runtime_overrides=overrides)
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    logger = configure_logging()
    device = resolve_device(args.device or config.get("device", "auto"))

    dataset_name = config["dataset"]["name"]
    train_history_len = int(checkpoint["config"]["dataset"]["history_len"])
    train_future_len = int(checkpoint["config"]["dataset"]["future_len"])
    eval_history_len = int(config["dataset"]["history_len"])
    eval_future_len = int(config["dataset"]["future_len"])
    split_files = restore_checkpoint_splits(checkpoint, config["dataset"]["data_dir"])
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
    amp_mode = config["evaluation"]["amp_mode"]
    with evaluation_autocast(device, amp_mode):
        bundle = collect_window_prediction_bundle(model, loader, normalizer, device=device)
    channel_names = infer_state_channel_names(dataset_name, state_dim=int(bundle["pred_phys"].shape[1]))
    summary = summarize_window_bundle(bundle, channel_names)
    artifact_stem = Path(str(Path(args.checkpoint).with_suffix("")) + ".window")
    output_dir = args.output_dir or (external_config or {}).get("output_dir")
    if output_dir is not None:
        artifact_stem = Path(output_dir) / artifact_stem.name
    artifact_stem.parent.mkdir(parents=True, exist_ok=True)
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
            "config_source": "checkpoint",
            "split_source": "checkpoint.split_manifest",
            "amp_mode": amp_mode,
        },
    )

    logger.info("Evaluation metrics: %s", summary["metrics"])
    logger.info("Detailed metric table:\n%s", format_metric_table(summary["metric_table"]))
    logger.info("Saved evaluation artifacts: %s", artifact_paths)


if __name__ == "__main__":
    main()
