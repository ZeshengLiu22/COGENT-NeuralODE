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
        continuous_cfg = config["model"]["continuous"]
        hidden_dim = int(continuous_cfg["hidden_dim"])
        num_layers = int(continuous_cfg.get("num_layers", 2))
        layer_type = str(continuous_cfg.get("gnn_type", "sage"))
        activation = str(continuous_cfg.get("activation", "softplus"))
        dropout = float(continuous_cfg.get("dropout", 0.0))
        self.use_history_in_ode = bool(model_cfg.get("use_history_in_ode", True))
        self.use_relative_time = bool(model_cfg.get("use_relative_time", True))
        self.relative_time_mode = str(model_cfg.get("relative_time_mode", "normalized")).lower()
        if self.relative_time_mode != "normalized":
            raise ValueError(f"Unsupported relative_time_mode: {self.relative_time_mode!r}")

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
        self.edge_index: torch.Tensor | None = None
        self.static_embed: torch.Tensor | None = None
        self.hist_context: torch.Tensor | None = None
        self.time_scale: torch.Tensor | float | None = None
        self.control = None

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
        self.control = control

    def _relative_time_feature(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.time_scale is None:
            raise RuntimeError("LatentNODEFunc relative-time scale has not been set.")
        t_tensor = t if torch.is_tensor(t) else torch.as_tensor(t, device=z.device, dtype=z.dtype)
        t_tensor = t_tensor.to(device=z.device, dtype=z.dtype)
        scale = self.time_scale if torch.is_tensor(self.time_scale) else torch.as_tensor(self.time_scale)
        scale = scale.to(device=z.device, dtype=z.dtype).clamp_min(torch.finfo(z.dtype).eps)
        rel_t = (t_tensor / scale).clamp(0.0, 1.0)
        return rel_t.reshape(1, 1).expand(z.shape[0], 1)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.edge_index is None or self.static_embed is None or self.control is None:
            raise RuntimeError("LatentNODEFunc context has not been set.")
        force_t = self.control.evaluate(t)
        input_parts = [z, force_t, self.static_embed]
        if self.use_history_in_ode:
            if self.hist_context is None:
                raise RuntimeError("LatentNODEFunc history context has not been set.")
            input_parts.append(self.hist_context)
        if self.use_relative_time:
            input_parts.append(self._relative_time_feature(t, z))
        inputs = torch.cat(input_parts, dim=-1)
        return self.net(inputs, self.edge_index)
