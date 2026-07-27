"""NODE1: state-space controlled graph Neural ODE."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("torchdiffeq is required for NODE models.") from exc

from models.common.interpolation import build_forcing_interpolant
from models.continuous.node_state_block import StateSpaceNODEFunc
from models.decoders.identity_decoder import IdentityDecoder
from models.encoders.history_encoder import HistoryEncoder


class NODE1Model(nn.Module):
    """State-space controlled graph NODE."""

    def __init__(self, num_static: int, num_force: int, num_state: int, config: dict) -> None:
        super().__init__()
        self.history_encoder = HistoryEncoder(num_static, num_force, num_state, config)
        self.dynamics = StateSpaceNODEFunc(
            state_dim=num_state,
            force_dim=num_force,
            static_dim=self.history_encoder.static_embed_dim,
            hist_dim=self.history_encoder.hist_context_dim,
            config=config,
        )
        self.decoder = IdentityDecoder()
        self.solver_cfg = config["solver"]

    def forward(self, data) -> torch.Tensor:
        encoded = self.history_encoder(data.x_static, data.state_hist, data.force_hist, data.edge_index)
        method = str(self.solver_cfg.get("ode_method", "midpoint"))
        rtol = float(self.solver_cfg.get("rtol", 1e-4))
        atol = float(self.solver_cfg.get("atol", 1e-5))
        use_adjoint = bool(self.solver_cfg.get("use_adjoint", False))
        interpolation = str(self.solver_cfg.get("interpolation", "linear"))
        options = self.solver_cfg.get("ode_options", None)
        solver = odeint_adjoint if use_adjoint else odeint

        with torch.autocast(device_type=data.x_static.device.type, enabled=False):
            y0 = encoded["last_state"].float()
            static_embed = encoded["static_embed"].float()
            hist_context = encoded["hist_context"].float()
            last_force = encoded["last_force"].float()
            force_future = data.force_future.float()
            t_future = data.t_future.float()

            control, eval_times = build_forcing_interpolant(last_force, force_future, t_future, method=interpolation)
            integration_times = torch.cat(
                [torch.zeros(1, device=eval_times.device, dtype=eval_times.dtype), eval_times],
                dim=0,
            )
            self.dynamics.set_context(data.edge_index, static_embed, hist_context, control)
            rollout = solver(self.dynamics, y0, integration_times, method=method, rtol=rtol, atol=atol, options=options)
            y_pred = rollout[1:].permute(1, 0, 2).contiguous()

        return self.decoder(y_pred)
