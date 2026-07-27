"""NODE2: latent-space controlled graph Neural ODE."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("torchdiffeq is required for NODE models.") from exc

from models.common.interpolation import build_forcing_interpolant
from models.common.mlp import MLP
from models.continuous.node_latent_block import LatentNODEFunc
from models.continuous.node_latent_block_structured import StructuredLatentNODEFunc
from models.decoders.mlp_decoder import MLPDecoder
from models.encoders.history_encoder import HistoryEncoder


class NODE2Model(nn.Module):
    """Latent-space controlled graph NODE."""

    def __init__(self, num_static: int, num_force: int, num_state: int, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        latent_dim = int(model_cfg["latent_dim"])
        decoder_hidden = list(model_cfg.get("decoder_hidden_dims", [latent_dim]))
        decoder_activation = str(model_cfg.get("decoder_activation", "gelu"))
        decoder_chunk_size = model_cfg.get("decoder_chunk_size", 262_144)
        dropout = float(model_cfg.get("dropout", 0.0))
        self.use_residual_decoder = bool(model_cfg.get("use_residual_decoder", True))

        self.history_encoder = HistoryEncoder(num_static, num_force, num_state, config)
        init_dim = self.history_encoder.hist_context_dim + num_state + self.history_encoder.static_embed_dim
        self.init_mlp = MLP(init_dim, [latent_dim], latent_dim, activation=decoder_activation, dropout=dropout)
        self.node2_vector_field_type = str(model_cfg.get("node2_vector_field_type", "v1_base")).lower()
        if self.node2_vector_field_type == "v1_base":
            dynamics_cls = LatentNODEFunc
        elif self.node2_vector_field_type == "structured_v2":
            dynamics_cls = StructuredLatentNODEFunc
        else:
            raise ValueError(
                "model.node2_vector_field_type must be 'v1_base' or 'structured_v2', "
                f"got {self.node2_vector_field_type!r}"
            )
        self.dynamics = dynamics_cls(
            latent_dim=latent_dim,
            force_dim=num_force,
            static_dim=self.history_encoder.static_embed_dim,
            hist_dim=self.history_encoder.hist_context_dim,
            config=config,
        )
        self.decoder = MLPDecoder(
            latent_dim,
            num_state,
            decoder_hidden,
            activation=decoder_activation,
            dropout=dropout,
            chunk_size=decoder_chunk_size,
        )
        self.solver_cfg = config["solver"]

    def forward(self, data) -> torch.Tensor:
        encoded = self.history_encoder(data.x_static, data.state_hist, data.force_hist, data.edge_index)
        init_inputs = torch.cat([encoded["hist_context"], encoded["last_state"], encoded["static_embed"]], dim=-1)
        method = str(self.solver_cfg.get("ode_method", "midpoint"))
        rtol = float(self.solver_cfg.get("rtol", 1e-4))
        atol = float(self.solver_cfg.get("atol", 1e-5))
        use_adjoint = bool(self.solver_cfg.get("use_adjoint", False))
        interpolation = str(self.solver_cfg.get("interpolation", "linear"))
        options = self.solver_cfg.get("ode_options", None)
        solver = odeint_adjoint if use_adjoint else odeint

        with torch.autocast(device_type=data.x_static.device.type, enabled=False):
            z0 = self.init_mlp(init_inputs.float())
            static_embed = encoded["static_embed"].float()
            last_force = encoded["last_force"].float()
            hist_context = encoded["hist_context"].float()
            force_future = data.force_future.float()
            t_future = data.t_future.float()

            control, eval_times = build_forcing_interpolant(last_force, force_future, t_future, method=interpolation)
            integration_times = torch.cat(
                [torch.zeros(1, device=eval_times.device, dtype=eval_times.dtype), eval_times],
                dim=0,
            )
            self.dynamics.set_context(
                data.edge_index,
                static_embed,
                control,
                hist_context=hist_context,
                time_scale=eval_times[-1],
            )
            rollout = solver(self.dynamics, z0, integration_times, method=method, rtol=rtol, atol=atol, options=options)
            z_future = rollout[1:].permute(1, 0, 2).contiguous()

        decoded = self.decoder(z_future)
        if self.use_residual_decoder:
            # Residual mode predicts normalized-space deltas from the last
            # observed state and broadcasts the anchor over future steps.
            decoded = encoded["last_state"].float().unsqueeze(1) + decoded
        return decoded
