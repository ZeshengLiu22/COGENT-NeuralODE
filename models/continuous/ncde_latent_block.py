"""Continuous block for NCDE latent evolution."""

from __future__ import annotations

import torch
from torch import nn

from models.common.gnn_blocks import GraphNetwork


class LatentNCDEFunc(nn.Module):
    """NCDE response ``f(z, s, G)`` returning ``[N, D_latent, D_control]``."""

    def __init__(self, latent_dim: int, static_dim: int, control_dim: int, config: dict) -> None:
        super().__init__()
        continuous_cfg = config["model"]["continuous"]
        hidden_dim = int(continuous_cfg["hidden_dim"])
        num_layers = int(continuous_cfg.get("num_layers", 2))
        layer_type = str(continuous_cfg.get("gnn_type", "sage"))
        activation = str(continuous_cfg.get("activation", "softplus"))
        dropout = float(continuous_cfg.get("dropout", 0.0))

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.net = GraphNetwork(
            input_dim=latent_dim + static_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim * control_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            activation=activation,
            dropout=dropout,
        )
        self.edge_index: torch.Tensor | None = None
        self.static_embed: torch.Tensor | None = None

    def set_context(self, edge_index: torch.Tensor, static_embed: torch.Tensor) -> None:
        self.edge_index = edge_index
        self.static_embed = static_embed

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        if self.edge_index is None or self.static_embed is None:
            raise RuntimeError("LatentNCDEFunc context has not been set.")
        inputs = torch.cat([z, self.static_embed], dim=-1)
        outputs = self.net(inputs, self.edge_index)
        return outputs.view(z.shape[0], self.latent_dim, self.control_dim)
