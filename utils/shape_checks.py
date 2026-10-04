"""Shape assertions used in critical tensor contracts."""

from __future__ import annotations

import torch


def ensure_shared_time_grid(t_future: torch.Tensor) -> torch.Tensor:
    """Return a single time grid, validating consistency across batched graphs."""

    if t_future.dim() == 1:
        return t_future
    if t_future.dim() != 2:
        raise ValueError(f"t_future must have rank 1 or 2, got shape {tuple(t_future.shape)}")

    ref = t_future[0]
    if not torch.allclose(t_future, ref.unsqueeze(0).expand_as(t_future)):
        raise ValueError(
            "Batched graphs with different time grids are not supported by the shared "
            "continuous solver path. Use batch_size=1 or align rollout times."
        )
    return ref
