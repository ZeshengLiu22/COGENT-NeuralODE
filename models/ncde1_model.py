"""NCDE1: latent-space controlled graph Neural CDE."""

from __future__ import annotations

import torch
from torch import nn

try:
    import torchcde
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("torchcde is required for NCDE models.") from exc

from models.common.interpolation import build_ncde_control_interpolant
from models.common.mlp import MLP
from models.continuous.ncde_latent_block import LatentNCDEFunc
from models.decoders.mlp_decoder import MLPDecoder
from models.encoders.history_encoder import HistoryEncoder


class NCDE1Model(nn.Module):
    """Latent-space controlled graph NCDE."""

    def __init__(self, num_static: int, num_force: int, num_state: int, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        latent_dim = int(model_cfg["latent_dim"])
        decoder_hidden = list(model_cfg.get("decoder_hidden_dims", [latent_dim]))
        decoder_activation = str(model_cfg.get("decoder_activation", "gelu"))
        decoder_chunk_size = model_cfg.get("decoder_chunk_size", 262_144)
        dropout = float(model_cfg.get("dropout", 0.0))

        self.history_encoder = HistoryEncoder(num_static, num_force, num_state, config)
        init_dim = self.history_encoder.hist_context_dim + num_state + self.history_encoder.static_embed_dim
        self.init_mlp = MLP(init_dim, [latent_dim], latent_dim, activation=decoder_activation, dropout=dropout)
        self.control_dim = num_force + 1
        self.dynamics = LatentNCDEFunc(
            latent_dim=latent_dim,
            static_dim=self.history_encoder.static_embed_dim,
            control_dim=self.control_dim,
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
        interpolation = str(self.solver_cfg.get("interpolation", "linear"))
        cde_method = str(self.solver_cfg.get("cde_method", "rk4"))
        rtol = float(self.solver_cfg.get("rtol", 1e-4))
        atol = float(self.solver_cfg.get("atol", 1e-5))
        use_adjoint = bool(self.solver_cfg.get("use_adjoint", False))
        options = self.solver_cfg.get("cde_options", None)

        with torch.autocast(device_type=data.x_static.device.type, enabled=False):
            z0 = self.init_mlp(init_inputs.float())
            last_force = encoded["last_force"].float()
            force_future = data.force_future.float()
            t_future = data.t_future.float()
            control, integration_times, control_dim = build_ncde_control_interpolant(
                last_force,
                force_future,
                t_future,
                method=interpolation,
            )
            if control_dim != self.control_dim:
                raise ValueError(f"NCDE control dimension mismatch: expected {self.control_dim}, got {control_dim}")

            self.dynamics.set_context(data.edge_index, encoded["static_embed"].float())
            rollout = torchcde.cdeint(
                X=control,
                z0=z0,
                func=self.dynamics,
                t=integration_times,
                method=cde_method,
                adjoint=use_adjoint,
                options=options,
                atol=atol,
                rtol=rtol,
            )
            z_future = rollout[1:].permute(1, 0, 2).contiguous()

        return self.decoder(z_future)
