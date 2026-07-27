"""Structured continuous block for NODE2 latent evolution."""

from __future__ import annotations

import torch
from torch import nn

from models.common.gnn_blocks import GraphNetwork
from models.common.mlp import MLP


_TERM_NAMES = ("local", "spatial", "forcing", "coupling")


def _make_term_norm(kind: str, latent_dim: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(latent_dim)
    if kind == "rmsnorm":
        return nn.RMSNorm(latent_dim)
    raise ValueError(
        "model.structured_dynamics.term_norm must be one of "
        "'none', 'layernorm', or 'rmsnorm', "
        f"got {kind!r}"
    )


class StructuredLatentNODEFunc(nn.Module):
    """Physics-inspired NODE2 vector field with additive latent mechanisms.

    The external solver contract matches ``LatentNODEFunc``:
    ``set_context(...)`` registers graph/history/control state once per
    rollout, then ``forward(t, z)`` returns ``dz/dt`` with shape
    ``[N_total, D_latent]``.
    """

    def __init__(self, latent_dim: int, force_dim: int, static_dim: int, hist_dim: int, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        continuous_cfg = model_cfg["continuous"]
        structured_cfg = model_cfg.get("structured_dynamics", {})

        hidden_dim = int(continuous_cfg["hidden_dim"])
        num_layers = int(continuous_cfg.get("num_layers", 2))
        layer_type = str(continuous_cfg.get("gnn_type", "sage"))
        activation = str(continuous_cfg.get("activation", "softplus"))
        dropout = float(continuous_cfg.get("dropout", 0.0))

        self.latent_dim = latent_dim
        self.force_dim = force_dim
        self.static_dim = static_dim
        self.hist_dim = hist_dim
        self.use_history_in_ode = bool(model_cfg.get("use_history_in_ode", True))
        self.use_relative_time = bool(model_cfg.get("use_relative_time", True))
        self.relative_time_mode = str(model_cfg.get("relative_time_mode", "normalized")).lower()
        if self.relative_time_mode != "normalized":
            raise ValueError(f"Unsupported relative_time_mode: {self.relative_time_mode!r}")

        self.use_f_local = bool(structured_cfg.get("use_f_local", True))
        self.use_f_spatial = bool(structured_cfg.get("use_f_spatial", True))
        self.use_f_forcing = bool(structured_cfg.get("use_f_forcing", True))
        self.use_f_coupling = bool(structured_cfg.get("use_f_coupling", True))
        self.term_norm_kind = str(structured_cfg.get("term_norm", "none")).lower()
        self.fusion = str(structured_cfg.get("fusion", "sum")).lower()
        self.gate_init = structured_cfg.get("gate_init", "active_mean")
        if self.fusion not in {"sum", "mean", "direct_gated", "softmax_gated"}:
            raise ValueError(
                "model.structured_dynamics.fusion must be one of "
                "'sum', 'mean', 'direct_gated', or 'softmax_gated', "
                f"got {self.fusion!r}"
            )
        if str(self.gate_init).lower() != "active_mean":
            raise ValueError(
                "model.structured_dynamics.gate_init currently supports only "
                f"'active_mean', got {self.gate_init!r}"
            )

        ctx_dim = static_dim
        if self.use_history_in_ode:
            ctx_dim += hist_dim
        if self.use_relative_time:
            ctx_dim += 1
        self.ctx_dim = ctx_dim

        local_hidden_dim = int(structured_cfg.get("local_hidden_dim", hidden_dim))
        spatial_out_hidden_dim = int(structured_cfg.get("spatial_out_hidden_dim", hidden_dim))
        forcing_hidden_dim = int(structured_cfg.get("forcing_hidden_dim", hidden_dim))
        coupling_hidden_dim = int(structured_cfg.get("coupling_hidden_dim", hidden_dim))

        self.local_mlp: MLP | None = None
        if self.use_f_local:
            self.local_mlp = MLP(
                latent_dim + ctx_dim,
                [local_hidden_dim],
                latent_dim,
                activation=activation,
                dropout=dropout,
            )

        self.spatial_gnn: GraphNetwork | None = None
        if self.use_f_spatial or self.use_f_coupling:
            self.spatial_gnn = GraphNetwork(
                input_dim=latent_dim + static_dim,
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers,
                layer_type=layer_type,
                activation=activation,
                dropout=dropout,
            )

        self.spatial_out_mlp: MLP | None = None
        if self.use_f_spatial:
            self.spatial_out_mlp = MLP(
                latent_dim + ctx_dim,
                [spatial_out_hidden_dim],
                latent_dim,
                activation=activation,
                dropout=dropout,
            )

        self.forcing_mlp: MLP | None = None
        if self.use_f_forcing:
            self.forcing_mlp = MLP(
                force_dim + ctx_dim,
                [forcing_hidden_dim],
                latent_dim,
                activation=activation,
                dropout=dropout,
            )

        self.coupling_mlp: MLP | None = None
        if self.use_f_coupling:
            self.coupling_mlp = MLP(
                latent_dim + force_dim + latent_dim + ctx_dim,
                [coupling_hidden_dim],
                latent_dim,
                activation=activation,
                dropout=dropout,
            )

        self.term_norms = nn.ModuleDict(
            {name: _make_term_norm(self.term_norm_kind, latent_dim) for name in _TERM_NAMES}
        )
        self.active_terms = {
            "local": self.use_f_local,
            "spatial": self.use_f_spatial,
            "forcing": self.use_f_forcing,
            "coupling": self.use_f_coupling,
        }
        self.num_active_terms = sum(int(active) for active in self.active_terms.values())
        self.register_buffer(
            "active_term_indices",
            torch.tensor(
                [index for index, name in enumerate(_TERM_NAMES) if self.active_terms[name]],
                dtype=torch.long,
            ),
            persistent=False,
        )

        if self.fusion == "direct_gated":
            gate_init = self._active_mean_gate_init()
            self.term_gates = nn.Parameter(gate_init)
            self.term_gate_logits = None
        elif self.fusion == "softmax_gated":
            self.term_gates = None
            self.term_gate_logits = nn.Parameter(torch.zeros(len(_TERM_NAMES), dtype=torch.float32))
        else:
            self.term_gates = None
            self.term_gate_logits = None

        self.edge_index: torch.Tensor | None = None
        self.static_embed: torch.Tensor | None = None
        self.hist_context: torch.Tensor | None = None
        self.time_scale: torch.Tensor | float | None = None
        self.control = None

    def _active_mean_gate_init(self) -> torch.Tensor:
        gates = torch.zeros(len(_TERM_NAMES), dtype=torch.float32)
        if self.num_active_terms == 0:
            return gates
        active_value = 1.0 / float(self.num_active_terms)
        for index, name in enumerate(_TERM_NAMES):
            if self.active_terms[name]:
                gates[index] = active_value
        return gates

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
            raise RuntimeError("StructuredLatentNODEFunc relative-time scale has not been set.")
        t_tensor = t if torch.is_tensor(t) else torch.as_tensor(t, device=z.device, dtype=z.dtype)
        t_tensor = t_tensor.to(device=z.device, dtype=z.dtype)
        scale = self.time_scale if torch.is_tensor(self.time_scale) else torch.as_tensor(self.time_scale)
        scale = scale.to(device=z.device, dtype=z.dtype).clamp_min(torch.finfo(z.dtype).eps)
        rel_t = (t_tensor / scale).clamp(0.0, 1.0)
        return rel_t.reshape(1, 1).expand(z.shape[0], 1)

    def _build_context(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.static_embed is None:
            raise RuntimeError("StructuredLatentNODEFunc static context has not been set.")
        ctx_parts = [self.static_embed]  # [N_total, D_static_embed]
        if self.use_history_in_ode:
            if self.hist_context is None:
                raise RuntimeError("StructuredLatentNODEFunc history context has not been set.")
            ctx_parts.append(self.hist_context)  # [N_total, D_hist]
        if self.use_relative_time:
            ctx_parts.append(self._relative_time_feature(t, z))  # [N_total, 1]
        return torch.cat(ctx_parts, dim=-1)  # [N_total, D_ctx]

    @staticmethod
    def _zero_like_latent(z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros(z.shape)

    def _fuse_terms(self, terms: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        active_names = [name for name in _TERM_NAMES if self.active_terms[name]]
        if not active_names:
            return self._zero_like_latent(z)

        active_terms = [self.term_norms[name](terms[name]) for name in active_names]
        if self.fusion == "sum":
            return torch.stack(active_terms, dim=0).sum(dim=0)
        if self.fusion == "mean":
            return torch.stack(active_terms, dim=0).mean(dim=0)

        active_indices = self.active_term_indices.to(device=z.device)
        if self.fusion == "direct_gated":
            if self.term_gates is None:
                raise RuntimeError("StructuredLatentNODEFunc direct gates were not initialized.")
            weights = self.term_gates.to(device=z.device, dtype=z.dtype).index_select(0, active_indices)
        elif self.fusion == "softmax_gated":
            if self.term_gate_logits is None:
                raise RuntimeError("StructuredLatentNODEFunc softmax gate logits were not initialized.")
            logits = self.term_gate_logits.to(device=z.device, dtype=z.dtype).index_select(0, active_indices)
            weights = torch.softmax(logits, dim=0)
        else:
            raise RuntimeError(f"Unsupported structured fusion mode: {self.fusion!r}")

        fused = self._zero_like_latent(z)
        for weight, term in zip(weights, active_terms, strict=True):
            fused = fused + weight * term
        return fused

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.edge_index is None or self.static_embed is None or self.control is None:
            raise RuntimeError("StructuredLatentNODEFunc context has not been set.")

        force_t = self.control.evaluate(t)  # [N_total, F_force]
        ctx = self._build_context(t, z)  # [N_total, D_ctx]

        spatial_feat: torch.Tensor | None = None
        if self.spatial_gnn is not None:
            spatial_inputs = torch.cat([z, self.static_embed], dim=-1)  # [N_total, D_latent + D_static_embed]
            spatial_feat = self.spatial_gnn(spatial_inputs, self.edge_index)  # [N_total, D_latent]

        if self.use_f_local:
            if self.local_mlp is None:
                raise RuntimeError("StructuredLatentNODEFunc local term was not initialized.")
            local_inputs = torch.cat([z, ctx], dim=-1)  # [N_total, D_latent + D_ctx]
            f_local = self.local_mlp(local_inputs)  # [N_total, D_latent]
        else:
            f_local = self._zero_like_latent(z)

        if self.use_f_spatial:
            if spatial_feat is None or self.spatial_out_mlp is None:
                raise RuntimeError("StructuredLatentNODEFunc spatial term was not initialized.")
            spatial_outputs = torch.cat([spatial_feat, ctx], dim=-1)  # [N_total, D_latent + D_ctx]
            f_spatial = self.spatial_out_mlp(spatial_outputs)  # [N_total, D_latent]
        else:
            f_spatial = self._zero_like_latent(z)

        if self.use_f_forcing:
            if self.forcing_mlp is None:
                raise RuntimeError("StructuredLatentNODEFunc forcing term was not initialized.")
            forcing_inputs = torch.cat([force_t, ctx], dim=-1)  # [N_total, F_force + D_ctx]
            f_forcing = self.forcing_mlp(forcing_inputs)  # [N_total, D_latent]
        else:
            f_forcing = self._zero_like_latent(z)

        if self.use_f_coupling:
            if spatial_feat is None or self.coupling_mlp is None:
                raise RuntimeError("StructuredLatentNODEFunc coupling term was not initialized.")
            coupling_inputs = torch.cat(
                [z, force_t, spatial_feat, ctx],
                dim=-1,
            )  # [N_total, D_latent + F_force + D_latent + D_ctx]
            f_coupling = self.coupling_mlp(coupling_inputs)  # [N_total, D_latent]
        else:
            f_coupling = self._zero_like_latent(z)

        return self._fuse_terms(
            {
                "local": f_local,
                "spatial": f_spatial,
                "forcing": f_forcing,
                "coupling": f_coupling,
            },
            z,
        )
