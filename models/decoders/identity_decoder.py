"""Identity decoder for NODE1."""

from __future__ import annotations

import torch
from torch import nn


class IdentityDecoder(nn.Module):
    """Return the input unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
