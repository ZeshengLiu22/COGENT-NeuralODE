"""Node-wise recurrent blocks."""

from __future__ import annotations

import torch
from torch import nn


class NodeWiseLSTM(nn.Module):
    """Apply an LSTM independently to each node's temporal embedding sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(x)
