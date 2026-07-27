"""Continuous block for NODE1 state-space evolution."""

from __future__ import annotations

import torch
from torch import nn

from models.common.gnn_blocks import GraphNetwork


class StateSpaceNODEFunc(nn.Module):
    """Derivative function ``dy/dt = f(y, u(t), s, c, G)``."""

    def __init__(self, state_dim: int, force_dim: int, static_dim: int, hist_dim: int, config: dict) -> None:
        super().__init__()
        continuous_cfg = config["model"]["continuous"]
        hidden_dim = int(continuous_cfg["hidden_dim"])
        num_layers = int(continuous_cfg.get("num_layers", 2))
        layer_type = str(continuous_cfg.get("gnn_type", "sage"))
        activation = str(continuous_cfg.get("activation", "softplus"))
        dropout = float(continuous_cfg.get("dropout", 0.0))

        self.net = GraphNetwork(
            input_dim=state_dim + force_dim + static_dim + hist_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            activation=activation,
            dropout=dropout,
        )
        self.edge_index: torch.Tensor | None = None
        self.static_embed: torch.Tensor | None = None
        self.hist_context: torch.Tensor | None = None
        self.control = None

    def set_context(
        self,
        edge_index: torch.Tensor,
        static_embed: torch.Tensor,
        hist_context: torch.Tensor,
        control,
    ) -> None:
        self.edge_index = edge_index
        self.static_embed = static_embed
        self.hist_context = hist_context
        self.control = control

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.edge_index is None or self.static_embed is None or self.hist_context is None or self.control is None:
            raise RuntimeError("StateSpaceNODEFunc context has not been set.")
        force_t = self.control.evaluate(t)
        inputs = torch.cat([y, force_t, self.static_embed, self.hist_context], dim=-1)
        return self.net(inputs, self.edge_index)
