"""Effective-horizon sampling with DDP synchronization."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.distributed as dist


def sample_effective_horizon(mode: str, k_min: int, k_max: int, device: torch.device) -> torch.Tensor:
    """Sample a single effective rollout horizon."""

    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, got {k_max} < {k_min}")

    if mode == "uniform_random":
        return torch.randint(k_min, k_max + 1, (1,), device=device, dtype=torch.long)
    if mode == "biased_long_horizon":
        horizons = torch.arange(k_min, k_max + 1, device=device, dtype=torch.float32)
        probs = horizons / horizons.sum()
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.to(dtype=torch.long) + k_min
    raise ValueError(f"Unsupported horizon sampling mode: {mode}")


def synchronized_horizon(mode: str, k_min: int, k_max: int, device: torch.device) -> int:
    """Sample ``k_eff`` once per global step and broadcast it across ranks."""

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            horizon = sample_effective_horizon(mode, k_min, k_max, device=device)
        else:
            horizon = torch.zeros(1, device=device, dtype=torch.long)
        dist.broadcast(horizon, src=0)
        return int(horizon.item())
    return int(sample_effective_horizon(mode, k_min, k_max, device=device).item())


def curriculum_horizon_max(
    *,
    epoch: int,
    k_min: int,
    target_k_max: int,
    enabled: bool,
    curriculum_epochs: int,
    warmup_fractions: Sequence[float],
) -> int:
    """Resolve the epoch-local maximum horizon for a staircase curriculum."""

    if epoch < 1:
        raise ValueError(f"epoch must be >= 1, got {epoch}")
    if target_k_max < k_min:
        raise ValueError(f"target_k_max must be >= k_min, got {target_k_max} < {k_min}")
    if not enabled or curriculum_epochs <= 0 or target_k_max == k_min:
        return int(target_k_max)

    fractions = [float(value) for value in warmup_fractions]
    if not fractions:
        raise ValueError("warmup_fractions must contain at least one value when horizon curriculum is enabled.")
    if any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError(f"warmup_fractions must be in (0, 1], got {fractions}")

    if epoch > curriculum_epochs:
        return int(target_k_max)

    stage_index = min((epoch - 1) * len(fractions) // curriculum_epochs, len(fractions) - 1)
    span = target_k_max - k_min
    cap = k_min + math.floor(span * fractions[stage_index] + 0.5)
    return int(max(k_min, min(target_k_max, cap)))
