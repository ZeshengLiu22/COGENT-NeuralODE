"""Configurable MLP helpers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Simple MLP with optional hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            is_last = index == len(dims) - 2
            if not is_last or activate_last:
                layers.append(get_activation(activation))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
