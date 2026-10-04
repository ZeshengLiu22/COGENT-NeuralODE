"""Metric helpers in physical units."""

from __future__ import annotations

import torch


def issm_speed_error_sums(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Physical speed SSE and count, derived from vx/vy (m/yr), for reporting only."""
    speed_pred = torch.linalg.vector_norm(pred[..., :2].double(), dim=-1)
    speed_true = torch.linalg.vector_norm(target[..., :2].double(), dim=-1)
    return torch.stack(((speed_pred - speed_true).square().sum(), speed_pred.new_tensor(speed_pred.numel())))


def add_issm_metrics(metrics: dict, speed_sums: torch.Tensor, prefix: str = "") -> None:
    """Name the paper-comparison physical metrics and include units in their keys."""
    metrics[f"{prefix}speed_rmse_m_per_yr"] = float(torch.sqrt(speed_sums[0] / speed_sums[1].clamp_min(1)))
    metrics[f"{prefix}thickness_rmse_m"] = metrics[f"{prefix}rmse_ch2"]


def channel_error_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-channel squared-error sums, absolute-error sums, and counts."""

    diff = pred - target
    sq_sum = (diff * diff).sum(dim=tuple(range(diff.dim() - 1)))
    abs_sum = diff.abs().sum(dim=tuple(range(diff.dim() - 1)))
    count_value = pred.numel() // pred.shape[-1]
    count = torch.full_like(sq_sum, float(count_value))
    return sq_sum, abs_sum, count


def overall_horizon_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizon-wise SSE and counts across nodes and channels."""

    if pred.dim() != 3:
        raise ValueError(f"Expected [N, K, F] tensor for horizon metrics, got {tuple(pred.shape)}")
    diff = pred - target
    sq_sum = (diff * diff).sum(dim=(0, 2))
    count = torch.full_like(sq_sum, float(pred.shape[0] * pred.shape[2]))
    return sq_sum, count


def summarize_channel_metrics(sq_sum: torch.Tensor, abs_sum: torch.Tensor, count: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Convert accumulated sums into RMSE/MAE metrics."""

    safe_count = torch.clamp(count, min=1.0)
    rmse = torch.sqrt(sq_sum / safe_count)
    mae = abs_sum / safe_count
    metrics = {
        f"{prefix}rmse": float(torch.sqrt(sq_sum.sum() / torch.clamp(count.sum(), min=1.0)).item()),
        f"{prefix}mae": float((abs_sum.sum() / torch.clamp(count.sum(), min=1.0)).item()),
    }
    for channel_index, value in enumerate(rmse.tolist()):
        metrics[f"{prefix}rmse_ch{channel_index}"] = float(value)
    for channel_index, value in enumerate(mae.tolist()):
        metrics[f"{prefix}mae_ch{channel_index}"] = float(value)
    return metrics


def summarize_horizon_curve(sq_sum: torch.Tensor, count: torch.Tensor) -> list[float]:
    """Return an overall horizon-wise RMSE curve."""

    safe_count = torch.clamp(count, min=1.0)
    return torch.sqrt(sq_sum / safe_count).cpu().tolist()
