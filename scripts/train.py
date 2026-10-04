#!/usr/bin/env python3
"""Training entrypoint for the unified continuous graph emulator."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.factory import build_dataset, build_loader, build_splits
from datasets.normalization import FeatureNormalizer
from models import build_model
from training import Trainer
from utils import cleanup_distributed, configure_logging, ensure_dir, load_config_bundle, save_json, seed_everything, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="YAML config files merged from left to right.")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--horizon-curriculum",
        choices=("on", "off"),
        default=None,
        help="Override training.train_horizon_curriculum.enabled from the command line.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.horizon_curriculum is None:
        return
    config["training"].setdefault("train_horizon_curriculum", {})["enabled"] = args.horizon_curriculum == "on"


def main() -> None:
    args = parse_args()
    config = load_config_bundle(args.config)
    apply_cli_overrides(config, args)
    ddp_info = setup_distributed(backend=config["distributed"].get("backend"))
    rank = int(ddp_info["rank"])
    device = torch.device("cuda", ddp_info["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(config["seed"]) + rank, deterministic=bool(config.get("deterministic", False)))

    dataset_name = config["dataset"]["name"]
    run_name = args.run_name or f"{dataset_name}_node2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / run_name)
    logger = configure_logging(output_dir / "train.log" if rank == 0 else None)
    if rank == 0:
        save_json(output_dir / "config.json", config)

    train_files, val_files, test_files = build_splits(config)
    if rank == 0:
        save_json(output_dir / "split_files.json", {"train": [str(p) for p in train_files], "val": [str(p) for p in val_files], "test": [str(p) for p in test_files]})

    train_dataset = build_dataset(dataset_name, train_files, split="train", config=config, normalizer=None, sample_windows=True)
    normalizer = FeatureNormalizer.fit_from_trajectories(train_dataset.iter_trajectories(), std_floor=float(config["normalization"]["std_floor"]))
    train_dataset.normalizer = normalizer
    val_dataset = build_dataset(dataset_name, val_files, split="val", config=config, normalizer=normalizer)
    test_dataset = build_dataset(dataset_name, test_files, split="test", config=config, normalizer=normalizer)

    dataset_cfg = config["dataset"]
    num_workers = int(dataset_cfg.get("num_workers", 0))
    val_num_workers = int(dataset_cfg.get("val_num_workers", num_workers))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    prefetch_factor = dataset_cfg.get("prefetch_factor", None)
    persistent_workers = bool(dataset_cfg.get("persistent_workers", False))
    if rank == 0:
        logger.info(
            "dataloader_settings=%s",
            {
                "num_workers": num_workers,
                "val_num_workers": val_num_workers,
                "pin_memory": pin_memory,
                "prefetch_factor": prefetch_factor,
                "persistent_workers": persistent_workers,
            },
        )

    train_loader = build_loader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=num_workers,
        distributed=bool(ddp_info["enabled"]),
        shuffle=True,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    eval_batch_size = int(config["evaluation"].get("batch_size", 1))
    val_loader = build_loader(
        val_dataset,
        batch_size=eval_batch_size,
        num_workers=val_num_workers,
        distributed=bool(ddp_info["enabled"]),
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=eval_batch_size,
        num_workers=val_num_workers,
        distributed=bool(ddp_info["enabled"]),
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    sample = train_dataset[0]
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1]).to(device)
    if ddp_info["enabled"]:
        model = DDP(model, device_ids=[ddp_info["local_rank"]] if device.type == "cuda" else None, find_unused_parameters=bool(config["distributed"].get("find_unused_parameters", False)))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        normalizer=normalizer,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    trainer.fit()
    cleanup_distributed()


if __name__ == "__main__":
    main()
