"""Train-split-only normalization with unit-independent constant detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass
class _RunningStats:
    """Accumulator for per-channel mean and variance."""

    mean: torch.Tensor
    m2: torch.Tensor
    count: int = 0

    def update(self, array: np.ndarray) -> None:
        values = torch.as_tensor(array, dtype=torch.float64)
        if values.dim() < 2:
            values = values.unsqueeze(-1)
        flat = values.reshape(-1, values.shape[-1])
        batch_count = int(flat.shape[0])
        if batch_count == 0:
            return
        batch_var, batch_mean = torch.var_mean(flat, dim=0, correction=0)
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_var * batch_count + delta.square() * (self.count * batch_count / total_count)
        self.mean += delta * (batch_count / total_count)
        self.count = total_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty running stats.")
        var = self.m2 / float(self.count)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        # A relative floating-point tolerance, not a cutoff in physical units.
        # Centered float64 accumulation preserves small real variances such as
        # rainfall in m/s, including channels whose std is far below 1e-5.
        constant = std <= torch.finfo(torch.float64).eps * self.mean.abs()
        safe_std = torch.where(constant, torch.ones_like(std), std)
        return self.mean, safe_std


class FeatureNormalizer:
    """Separate normalizers for static, forcing, and state channels."""

    def __init__(
        self,
        static_mean: torch.Tensor,
        static_std: torch.Tensor,
        force_mean: torch.Tensor,
        force_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        std_floor: float = 1e-5,
    ) -> None:
        self.static_mean = static_mean.float()
        self.static_std = static_std.float()
        self.force_mean = force_mean.float()
        self.force_std = force_std.float()
        self.state_mean = state_mean.float()
        self.state_std = state_std.float()
        # Retained as serialized legacy metadata; never a physical-scale cutoff.
        self.std_floor = float(std_floor)

    @classmethod
    def fit_from_trajectories(cls, trajectories: Iterable[Any], std_floor: float = 1e-5) -> "FeatureNormalizer":
        """Estimate float64 population statistics from training trajectories.

        ``std_floor`` is accepted for existing config/checkpoint metadata only;
        constant detection uses float64 relative precision instead.
        """

        trajectories = list(trajectories)
        if not trajectories:
            raise ValueError("Need at least one trajectory to fit a normalizer.")

        static_dim = trajectories[0].x_static.shape[-1]
        force_dim = trajectories[0].force.shape[-1]
        state_dim = trajectories[0].state.shape[-1]

        static_stats = _RunningStats(torch.zeros(static_dim, dtype=torch.float64), torch.zeros(static_dim, dtype=torch.float64))
        force_stats = _RunningStats(torch.zeros(force_dim, dtype=torch.float64), torch.zeros(force_dim, dtype=torch.float64))
        state_stats = _RunningStats(torch.zeros(state_dim, dtype=torch.float64), torch.zeros(state_dim, dtype=torch.float64))

        for trajectory in trajectories:
            static_stats.update(trajectory.x_static)
            force_stats.update(trajectory.force)
            state_stats.update(trajectory.state)

        static_mean, static_std = static_stats.finalize()
        force_mean, force_std = force_stats.finalize()
        state_mean, state_std = state_stats.finalize()
        return cls(static_mean, static_std, force_mean, force_std, state_mean, state_std, std_floor=std_floor)

    def transform_static(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize static features."""

        return self._transform(tensor, self.static_mean, self.static_std)

    def transform_force(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize forcing features."""

        return self._transform(tensor, self.force_mean, self.force_std)

    def transform_state(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize state features."""

        return self._transform(tensor, self.state_mean, self.state_std)

    def inverse_state(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map normalized state tensors back to physical units."""

        mean = self.state_mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.state_std.to(device=tensor.device, dtype=tensor.dtype)
        return tensor * std + mean

    def state_stats(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state mean/std on the requested device."""

        if device is None:
            return self.state_mean, self.state_std
        return self.state_mean.to(device), self.state_std.to(device)

    def to_dict(self) -> dict[str, list[float] | float]:
        """Serialize the normalizer for checkpointing."""

        return {
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "force_mean": self.force_mean.tolist(),
            "force_std": self.force_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "std_floor": self.std_floor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float] | float]) -> "FeatureNormalizer":
        """Restore a normalizer from a serialized dictionary."""

        return cls(
            static_mean=torch.tensor(payload["static_mean"], dtype=torch.float32),
            static_std=torch.tensor(payload["static_std"], dtype=torch.float32),
            force_mean=torch.tensor(payload["force_mean"], dtype=torch.float32),
            force_std=torch.tensor(payload["force_std"], dtype=torch.float32),
            state_mean=torch.tensor(payload["state_mean"], dtype=torch.float32),
            state_std=torch.tensor(payload["state_std"], dtype=torch.float32),
            std_floor=float(payload["std_floor"]),
        )

    @staticmethod
    def _transform(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        mean = mean.to(device=tensor.device, dtype=tensor.dtype)
        std = std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std
