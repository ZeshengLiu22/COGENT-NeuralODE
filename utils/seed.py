"""Randomness and distributed-runtime helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return ``True`` when torch distributed is initialized."""

    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current distributed rank, defaulting to zero."""

    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the distributed world size, defaulting to one."""

    if is_distributed():
        return dist.get_world_size()
    return 1


def setup_distributed(backend: Optional[str] = None) -> dict[str, int | bool]:
    """Initialize DDP from ``torchrun`` environment variables when present."""

    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return {
            "enabled": True,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": local_rank,
        }

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def cleanup_distributed() -> None:
    """Destroy the distributed process group when initialized."""

    if is_distributed():
        dist.destroy_process_group()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
