"""Shared dataset, split, and loader construction for entrypoints."""

import logging
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from .adcirc_dataset import ADCIRCDataset
from .anuga_dataset import ANUGADataset
from .issm_dataset import ISSMDataset
from .split_utils import discover_files, issm_rate_modulo_split, random_split


DATASET_REGISTRY = {"anuga": ANUGADataset, "adcirc": ADCIRCDataset, "issm": ISSMDataset}


class DistributedEvaluationSampler(Sampler[int]):
    """Shard evaluation indices exactly once globally, without padding."""

    def __init__(self, dataset, num_replicas: int | None = None, rank: int | None = None) -> None:
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else int(num_replicas)
        self.rank = dist.get_rank() if rank is None else int(rank)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Distributed evaluation requires num_replicas >= 1 and 0 <= rank < num_replicas.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.num_replicas))


def build_splits(config: dict) -> tuple[list[Path], list[Path], list[Path]]:
    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]
    files = discover_files(dataset_cfg["data_dir"], dataset_cfg["file_patterns"])
    strategy = split_cfg["strategy"]
    if strategy == "random":
        return random_split(
            files,
            train=split_cfg["train"],
            val=split_cfg["val"],
            test=split_cfg["test"],
            seed=split_cfg.get("seed", config["seed"]),
        )
    if strategy == "issm_rate_modulo":
        return issm_rate_modulo_split(
            files,
            modulo=split_cfg.get("modulo", 20),
            val_remainder=split_cfg.get("val_remainder", 0),
            test_remainder=split_cfg.get("test_remainder", 10),
        )
    raise ValueError(f"Unknown split strategy: {strategy}")


def build_dataset(
    dataset_name: str,
    files: list[Path],
    split: str,
    config: dict,
    normalizer=None,
    *,
    history_len: int | None = None,
    future_len: int | None = None,
    sample_windows: bool = False,
):
    dataset_cfg = config["dataset"]
    sampling_cfg = dataset_cfg.get("sampled_windows", {}) if sample_windows else {}
    return DATASET_REGISTRY[dataset_name](
        scenario_files=files,
        history_len=dataset_cfg["history_len"] if history_len is None else history_len,
        future_len=dataset_cfg["future_len"] if future_len is None else future_len,
        split=split,
        stride=dataset_cfg.get("stride", 1),
        normalizer=normalizer,
        cache_in_memory=dataset_cfg.get("cache_in_memory", False),
        epoch_num_windows=sampling_cfg.get("epoch_num_windows"),
        windows_per_scenario=sampling_cfg.get("windows_per_scenario"),
        seed=dataset_cfg.get("seed", config["seed"]),
        adapter_kwargs=dataset_cfg.get(dataset_name, {}),
        window_reference=dataset_cfg.get("window_reference"),
    )


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True) if shuffle else DistributedEvaluationSampler(dataset)
    resamples_each_epoch = (
        getattr(dataset, "windows_per_scenario", None) is not None
        or getattr(dataset, "epoch_num_windows", None) is not None
    )
    if getattr(dataset, "split", None) == "train" and resamples_each_epoch and num_workers > 0 and persistent_workers:
        logging.getLogger(__name__).warning(
            "Disabling persistent_workers for the training loader because epoch window resampling "
            "requires workers to receive the updated window list."
        )
        persistent_workers = False
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)
