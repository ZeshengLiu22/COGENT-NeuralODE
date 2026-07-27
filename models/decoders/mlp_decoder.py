"""Small per-node decoder MLP."""

from __future__ import annotations

import torch
from torch import nn

from models.common.mlp import MLP


class MLPDecoder(nn.Module):
    """Decode nodewise latent states into physical states."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_dims: list[int],
        activation: str = "gelu",
        dropout: float = 0.0,
        chunk_size: int | None = 262_144,
    ) -> None:
        super().__init__()
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("decoder chunk_size must be >= 1 or None")
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.decoder = MLP(latent_dim, hidden_dims, state_dim, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self._decode_flat(x)
        if x.dim() != 3:
            raise ValueError(f"Expected rank-2 or rank-3 latent tensor, got shape {tuple(x.shape)}")
        batch, steps, channels = x.shape
        flat = x.reshape(batch * steps, channels)
        decoded = self._decode_flat(flat)
        return decoded.view(batch, steps, -1)

    def _decode_flat(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return torch.cat([self.decoder(chunk) for chunk in x.split(self.chunk_size, dim=0)], dim=0)
        return self.decoder(x)
