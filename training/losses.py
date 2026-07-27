"""Version-1 rollout losses."""

from __future__ import annotations

import torch


def rollout_mse(y_pred: torch.Tensor, y_true: torch.Tensor, k_eff: int | None = None) -> torch.Tensor:
    """Compute rollout MSE over the first ``k_eff`` steps when provided."""

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch in rollout_mse: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}")
    if k_eff is not None:
        y_pred = y_pred[:, :k_eff, :]
        y_true = y_true[:, :k_eff, :]
    return torch.mean((y_pred - y_true) ** 2)
