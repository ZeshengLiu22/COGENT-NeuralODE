"""Graph neural network blocks used throughout the models."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv

from .mlp import get_activation


def _build_conv(layer_type: str, in_dim: int, out_dim: int):
    layer_type = layer_type.lower()
    if layer_type == "sage":
        return SAGEConv(in_dim, out_dim)
    if layer_type == "gcn":
        return GCNConv(in_dim, out_dim)
    if layer_type == "graphconv":
        return GraphConv(in_dim, out_dim)
    raise ValueError(f"Unsupported GNN layer type: {layer_type}")


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
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [input_dim]
        if num_layers == 1:
            dims.append(output_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(output_dim)

        self.convs = nn.ModuleList(
            _build_conv(layer_type, dims[index], dims[index + 1]) for index in range(len(dims) - 1)
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
