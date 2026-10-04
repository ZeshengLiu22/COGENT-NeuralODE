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
        self.chunk_size = chunk_size
        self.decoder = MLP(latent_dim, hidden_dims, state_dim, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        decoded = self._decode_flat(flat)
        return decoded.reshape(*x.shape[:-1], decoded.shape[-1])

    def _decode_flat(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return torch.cat([self.decoder(chunk) for chunk in x.split(self.chunk_size, dim=0)], dim=0)
        return self.decoder(x)
