"""Continuous block for NODE2 latent evolution."""

from __future__ import annotations

import torch
from torch import nn

from models.common.gnn_blocks import GraphNetwork


class LatentNODEFunc(nn.Module):
    """Derivative function ``dz/dt = f(z, u(t), s, c_hist, rel_t, G)``.

    ``c_hist`` and ``rel_t`` are optional upgrade-v1 inputs.  When disabled,
    the original NODE2 vector field ``f(z, u(t), s, G)`` is recovered.
    """

    def __init__(self, latent_dim: int, force_dim: int, static_dim: int, hist_dim: int, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        continuous_cfg = model_cfg["continuous"]
        hidden_dim = continuous_cfg["hidden_dim"]
        num_layers = continuous_cfg.get("num_layers", 2)
        layer_type = continuous_cfg.get("gnn_type", "sage")
        activation = continuous_cfg.get("activation", "softplus")
        dropout = continuous_cfg.get("dropout", 0.0)
        self.use_history_in_ode = model_cfg.get("use_history_in_ode", True)
        self.use_relative_time = model_cfg.get("use_relative_time", True)

        input_dim = latent_dim + force_dim + static_dim
        if self.use_history_in_ode:
            input_dim += hist_dim
        if self.use_relative_time:
            input_dim += 1

        self.net = GraphNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            activation=activation,
            dropout=dropout,
        )

    def set_context(
        self,
        edge_index: torch.Tensor,
        static_embed: torch.Tensor,
        control,
        hist_context: torch.Tensor | None = None,
        time_scale: torch.Tensor | float | None = None,
    ) -> None:
        self.edge_index = edge_index
        self.static_embed = static_embed
        self.hist_context = hist_context
        self.time_scale = time_scale
        # Forcing is per-batch runtime data, not a checkpointed submodule.
        self.evaluate_forcing = control.evaluate

    def _relative_time_feature(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_tensor = t.to(device=z.device, dtype=z.dtype)
        scale = torch.as_tensor(self.time_scale, device=z.device, dtype=z.dtype).clamp_min(torch.finfo(z.dtype).eps)
        rel_t = (t_tensor / scale).clamp(0.0, 1.0)
        return rel_t.reshape(1, 1).expand(z.shape[0], 1)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        force_t = self.evaluate_forcing(t)
        input_parts = [z, force_t, self.static_embed]
        if self.use_history_in_ode:
            input_parts.append(self.hist_context)
        if self.use_relative_time:
            input_parts.append(self._relative_time_feature(t, z))
        inputs = torch.cat(input_parts, dim=-1)
        return self.net(inputs, self.edge_index)
