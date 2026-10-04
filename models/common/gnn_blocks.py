"""Graph neural network blocks used throughout the models."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv

from .mlp import get_activation


class GraphNetwork(nn.Module):
    """Stack of message-passing layers with configurable hidden size."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "sage",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        conv_cls = {"sage": SAGEConv, "gcn": GCNConv, "graphconv": GraphConv}[layer_type]
        self.convs = nn.ModuleList(
            conv_cls(dims[index], dims[index + 1]) for index in range(num_layers)
        )
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index != len(self.convs) - 1:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x
