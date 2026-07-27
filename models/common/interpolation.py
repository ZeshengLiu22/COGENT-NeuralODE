"""Interpolation utilities for NODE and NCDE control paths."""

from __future__ import annotations

import torch
import torchcde

from utils.shape_checks import ensure_shared_time_grid


def _build_interpolator(values: torch.Tensor, t_control: torch.Tensor, method: str):
    method = method.lower()
    if method == "linear":
        coeffs = torchcde.linear_interpolation_coeffs(values, t=t_control)
        return torchcde.LinearInterpolation(coeffs, t=t_control)
    if method == "hermite_cubic_backward":
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values, t=t_control)
        return torchcde.CubicSpline(coeffs, t=t_control)
    raise ValueError(f"Unsupported interpolation method: {method}")


def build_forcing_interpolant(
    last_force: torch.Tensor,
    force_future: torch.Tensor,
    t_future: torch.Tensor,
    method: str = "linear",
):
    """Build an interpolant for NODE forcing evaluation."""

    shared_t = ensure_shared_time_grid(t_future).to(device=force_future.device, dtype=torch.float32)
    zero = torch.zeros(1, device=force_future.device, dtype=torch.float32)
    t_control = torch.cat([zero, shared_t], dim=0)
    values = torch.cat([last_force.unsqueeze(1), force_future], dim=1).float()
    return _build_interpolator(values, t_control, method=method), shared_t


def build_ncde_control_interpolant(
    last_force: torch.Tensor,
    force_future: torch.Tensor,
    t_future: torch.Tensor,
    method: str = "linear",
):
    """Build an NCDE control path using forcing and relative time."""

    shared_t = ensure_shared_time_grid(t_future).to(device=force_future.device, dtype=torch.float32)
    zero = torch.zeros(1, device=force_future.device, dtype=torch.float32)
    t_control = torch.cat([zero, shared_t], dim=0)
    force_values = torch.cat([last_force.unsqueeze(1), force_future], dim=1).float()
    time_channel = t_control.view(1, -1, 1).expand(force_values.shape[0], -1, 1)
    control_values = torch.cat([force_values, time_channel], dim=-1)
    return _build_interpolator(control_values, t_control, method=method), t_control, control_values.shape[-1]
