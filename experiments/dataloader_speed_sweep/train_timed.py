#!/usr/bin/env python3
"""Timed ISSM training entrypoint for dataloader-speed experiments."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.factory import build_dataset, build_loader, build_splits
from datasets.normalization import FeatureNormalizer
from datasets.split_utils import make_split_manifest
from models import build_model
from training import Trainer
from utils import cleanup_distributed, configure_logging, ensure_dir, load_config_bundle, save_json, seed_everything, setup_distributed
from utils.logging_utils import rank0_log
from utils.seed import get_rank


T = TypeVar("T")


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


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.horizon_curriculum is None:
        return
    config["training"].setdefault("train_horizon_curriculum", {})["enabled"] = args.horizon_curriculum == "on"


class TimedTrainer(Trainer):
    """Trainer variant that records epoch wall times and skips final test eval."""

    def _sync_for_timing(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _timed(self, fn: Callable[[], T]) -> tuple[T, float]:
        self._sync_for_timing()
        start = time.perf_counter()
        result = fn()
        self._sync_for_timing()
        elapsed = torch.tensor([time.perf_counter() - start], device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return result, float(elapsed.item())

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_dataset.set_epoch(epoch)

            train_windows = len(self.train_dataset)
            train_batches_per_rank = len(self.train_loader)
            train_metrics, train_seconds = self._timed(lambda: self.train_epoch(epoch))

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "timing": {
                    "train_seconds": train_seconds,
                    "train_windows": train_windows,
                    "train_batches_per_rank": train_batches_per_rank,
                    "train_windows_per_sec": train_windows / max(train_seconds, 1.0e-12),
                },
            }

            if epoch % self.val_every == 0 or epoch == self.epochs:
                val_rollout, rollout_seconds = self._timed(
                    lambda: self.evaluator.evaluate_full_rollout(
                        self.val_dataset, known_steps=self.known_steps,
                    )
                )
                epoch_record["val_rollout"] = val_rollout
                epoch_record["timing"]["val_rollout_seconds"] = rollout_seconds

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(epoch_record)
            if get_rank() == 0:
                parts = [
                    f"Epoch {epoch:03d}",
                    f"train_seconds={train_seconds:.3f}",
                    f"train_windows_per_sec={epoch_record['timing']['train_windows_per_sec']:.3f}",
                    f"train_norm_mse={train_metrics['train_norm_mse']:.6f}",
                ]
                if "val_rollout_seconds" in epoch_record["timing"]:
                    parts.append(f"val_rollout_seconds={epoch_record['timing']['val_rollout_seconds']:.3f}")
                rank0_log(self.logger, " | ".join(parts))
                save_json(self.output_dir / "history.json", history)

        summary = self._summarize_speed(history)
        if get_rank() == 0:
            save_json(self.output_dir / "speed_summary.json", summary)
        return summary

    def _summarize_speed(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        train_times = [float(row["timing"]["train_seconds"]) for row in history]
        train_rates = [float(row["timing"]["train_windows_per_sec"]) for row in history]
        warm_slice = slice(1, None) if len(history) > 1 else slice(None)
        warm_train_times = train_times[warm_slice]
        warm_train_rates = train_rates[warm_slice]

        total_epoch_times = [
            float(row["timing"]["train_seconds"])
            + float(row["timing"].get("val_rollout_seconds", 0.0))
            for row in history
        ]
        rollout_times = [float(row["timing"]["val_rollout_seconds"]) for row in history if "val_rollout_seconds" in row["timing"]]
        return {
            "epochs": len(history),
            "mean_train_seconds_all": sum(train_times) / max(len(train_times), 1),
            "mean_train_seconds_warm": sum(warm_train_times) / max(len(warm_train_times), 1),
            "mean_train_windows_per_sec_all": sum(train_rates) / max(len(train_rates), 1),
            "mean_train_windows_per_sec_warm": sum(warm_train_rates) / max(len(warm_train_rates), 1),
            "mean_val_rollout_seconds_all": (sum(rollout_times) / len(rollout_times)) if rollout_times else None,
            "mean_epoch_seconds_all": sum(total_epoch_times) / max(len(total_epoch_times), 1),
            "history": history,
        }


def main() -> None:
    args = parse_args()
    config = load_config_bundle(args.config)
    apply_cli_overrides(config, args)
    ddp_info = setup_distributed(backend=config.get("distributed", {}).get("backend"))
    rank = int(ddp_info["rank"])
    device = torch.device("cuda", ddp_info["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(config["seed"]) + rank, deterministic=bool(config.get("deterministic", False)))

    dataset_name = config["dataset"]["name"]
    run_name = args.run_name or f"{dataset_name}_node2_speed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / run_name)
    logger = configure_logging(output_dir / "train.log" if rank == 0 else None)
    if rank == 0:
        save_json(output_dir / "config.json", config)
        (output_dir / "config_stack.txt").write_text("".join(f"{path}\n" for path in args.config), encoding="utf-8")

    train_files, val_files, test_files = build_splits(config)
    if rank == 0:
        save_json(
            output_dir / "split_files.json",
            make_split_manifest({"train": train_files, "val": val_files, "test": test_files}, config["dataset"]["data_dir"]),
        )

    train_dataset = build_dataset(dataset_name, train_files, split="train", config=config, normalizer=None, sample_windows=True)
    normalizer = FeatureNormalizer.fit_from_trajectories(
        train_dataset.iter_trajectories(),
        std_floor=float(config["normalization"]["std_floor"]),
    )
    train_dataset.normalizer = normalizer
    val_dataset = build_dataset(dataset_name, val_files, split="val", config=config, normalizer=normalizer, sample_windows=False)
    test_dataset = build_dataset(dataset_name, test_files, split="test", config=config, normalizer=normalizer, sample_windows=False)

    dataset_cfg = config["dataset"]
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    prefetch_factor = dataset_cfg.get("prefetch_factor", None)
    persistent_workers = bool(dataset_cfg.get("persistent_workers", False))

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
    if rank == 0:
        rank0_log(
            logger,
            "dataloader_settings="
            f"num_workers={num_workers}, "
            f"pin_memory={pin_memory}, prefetch_factor={prefetch_factor}, "
            f"persistent_workers={persistent_workers}",
        )

    sample = train_dataset[0]
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1]).to(device)
    if ddp_info["enabled"]:
        model = DDP(
            model,
            device_ids=[ddp_info["local_rank"]] if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("distributed", {}).get("find_unused_parameters", False)),
        )

    trainer = TimedTrainer(
        model=model,
        train_loader=train_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        normalizer=normalizer,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
