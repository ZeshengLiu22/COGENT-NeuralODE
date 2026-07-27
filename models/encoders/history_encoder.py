"""Shared history encoder used by all baselines."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn

from models.common.gnn_blocks import GraphNetwork
from models.common.lstm_blocks import NodeWiseLSTM
from models.common.mlp import MLP, get_activation
from utils.shape_checks import assert_rank


def _math_sdp_context(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)


class HistoryEncoder(nn.Module):
    """Static MLP + per-step GNN + temporal history summarizer.

    The public tensor contract stays node-major for PyG batches:
    ``step_embeds`` is ``[N_total, H, D_enc]`` and ``hist_context`` is
    ``[N_total, D_hist]``.  The default temporal summarizer is a
    temporal-only Transformer over the history dimension for each node
    independently; the previous node-wise LSTM remains available.
    """

    def __init__(self, num_static: int, num_force: int, num_state: int, config: dict) -> None:
        super().__init__()
        history_cfg = config["model"]["history_encoder"]
        static_hidden = history_cfg.get("static_hidden_dims", [64])
        static_embed_dim = int(history_cfg["static_embed_dim"])
        gnn_hidden_dim = int(history_cfg["gnn_hidden_dim"])
        gnn_layers = int(history_cfg.get("gnn_num_layers", 2))
        gnn_type = str(history_cfg.get("gnn_type", "sage"))
        gnn_activation = str(history_cfg.get("gnn_activation", "gelu"))
        gnn_dropout = float(history_cfg.get("dropout", 0.0))
        lstm_hidden_dim = int(history_cfg["lstm_hidden_dim"])
        lstm_layers = int(history_cfg.get("lstm_num_layers", 1))
        encoder_type = str(history_cfg.get("history_encoder_type", "transformer")).lower()
        if encoder_type not in {"transformer", "lstm"}:
            raise ValueError(f"history_encoder_type must be 'transformer' or 'lstm', got {encoder_type!r}")
        self.use_transformer_history = bool(history_cfg.get("use_transformer_history", True)) and encoder_type == "transformer"
        self.history_context_pooling = str(history_cfg.get("history_context_pooling", "last")).lower()
        if self.history_context_pooling not in {"last", "mean"}:
            raise ValueError(
                "history_context_pooling must be 'last' or 'mean', "
                f"got {self.history_context_pooling!r}"
            )
        self.history_use_positional_encoding = bool(history_cfg.get("history_use_positional_encoding", True))
        transformer_chunk_size = history_cfg.get("history_transformer_chunk_size", 8192)
        self.history_transformer_chunk_size = None if transformer_chunk_size is None else int(transformer_chunk_size)
        if self.history_transformer_chunk_size is not None and self.history_transformer_chunk_size < 1:
            raise ValueError("history_transformer_chunk_size must be >= 1 or null")
        self.history_transformer_force_math_sdp = bool(history_cfg.get("history_transformer_force_math_sdp", True))

        self.state_dim = num_state
        self.force_dim = num_force
        self.static_embed_dim = static_embed_dim
        self.hist_context_dim = lstm_hidden_dim

        self.static_mlp = MLP(num_static, static_hidden, static_embed_dim, activation=gnn_activation, dropout=gnn_dropout)
        self.step_gnn = GraphNetwork(
            input_dim=num_state + num_force + static_embed_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=static_embed_dim,
            num_layers=gnn_layers,
            layer_type=gnn_type,
            activation=gnn_activation,
            dropout=gnn_dropout,
        )
        self.temporal_lstm: NodeWiseLSTM | None = None
        self.temporal_transformer: nn.TransformerEncoder | None = None
        self.transformer_context_projection: nn.Module | None = None
        if self.use_transformer_history:
            transformer_layers = int(history_cfg.get("history_transformer_num_layers", 2))
            transformer_heads = int(history_cfg.get("history_transformer_num_heads", 4))
            transformer_ff_dim = int(history_cfg.get("history_transformer_ff_dim", static_embed_dim * 4))
            transformer_dropout = float(history_cfg.get("history_transformer_dropout", gnn_dropout))
            if transformer_layers < 1:
                raise ValueError("history_transformer_num_layers must be >= 1")
            if transformer_heads < 1:
                raise ValueError("history_transformer_num_heads must be >= 1")
            if static_embed_dim % transformer_heads != 0:
                raise ValueError(
                    "history_transformer_num_heads must divide static_embed_dim; "
                    f"got static_embed_dim={static_embed_dim}, heads={transformer_heads}"
                )
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=static_embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation=get_activation(gnn_activation),
                batch_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)
            if static_embed_dim == lstm_hidden_dim:
                self.transformer_context_projection = nn.Identity()
            else:
                self.transformer_context_projection = nn.Linear(static_embed_dim, lstm_hidden_dim)
        else:
            self.temporal_lstm = NodeWiseLSTM(
                static_embed_dim,
                lstm_hidden_dim,
                num_layers=lstm_layers,
                dropout=gnn_dropout,
            )

    @staticmethod
    def _sinusoidal_positional_encoding(
        length: int,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[1, H, D]`` sinusoidal positions for temporal-only attention."""

        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / channels)
        )
        encoding = torch.zeros(length, channels, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        if channels > 1:
            encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0).to(dtype=dtype)

    def forward(
        self,
        x_static: torch.Tensor,
        state_hist: torch.Tensor,
        force_hist: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert_rank(x_static, 2, "x_static")
        assert_rank(state_hist, 3, "state_hist")
        assert_rank(force_hist, 3, "force_hist")

        static_embed = self.static_mlp(x_static)
        history_len = state_hist.shape[1]
        step_embeds: list[torch.Tensor] = []

        for step in range(history_len):
            step_inputs = torch.cat([state_hist[:, step, :], force_hist[:, step, :], static_embed], dim=-1)
            step_embeds.append(self.step_gnn(step_inputs, edge_index))

        # Node-major PyG batching: temporal modules see each node as one
        # independent sequence of H per-step spatial embeddings.
        stacked = torch.stack(step_embeds, dim=1)
        if self.use_transformer_history:
            if self.temporal_transformer is None or self.transformer_context_projection is None:
                raise RuntimeError("Transformer history encoder was not initialized.")
            temporal_inputs = stacked
            if self.history_use_positional_encoding:
                temporal_inputs = temporal_inputs + self._sinusoidal_positional_encoding(
                    history_len,
                    stacked.shape[-1],
                    device=stacked.device,
                    dtype=stacked.dtype,
                )
            temporal_outputs = self._run_temporal_transformer(temporal_inputs)
            if self.history_context_pooling == "mean":
                pooled_context = temporal_outputs.mean(dim=1)
            else:
                pooled_context = temporal_outputs[:, -1, :]
            hist_context = self.transformer_context_projection(pooled_context)
        else:
            if self.temporal_lstm is None:
                raise RuntimeError("LSTM history encoder was not initialized.")
            _, (hidden, _) = self.temporal_lstm(stacked)
            hist_context = hidden[-1]

        return {
            "static_embed": static_embed,
            "step_embeds": stacked,
            "hist_context": hist_context,
            "last_state": state_hist[:, -1, :],
            "last_force": force_hist[:, -1, :],
        }

    def _run_temporal_transformer(self, temporal_inputs: torch.Tensor) -> torch.Tensor:
        if self.temporal_transformer is None:
            raise RuntimeError("Transformer history encoder was not initialized.")

        chunk_size = self.history_transformer_chunk_size
        force_math_sdp = temporal_inputs.is_cuda and self.history_transformer_force_math_sdp
        with _math_sdp_context(force_math_sdp):
            if chunk_size is None or temporal_inputs.shape[0] <= chunk_size:
                return self.temporal_transformer(temporal_inputs)
            return torch.cat(
                [self.temporal_transformer(chunk) for chunk in temporal_inputs.split(chunk_size, dim=0)],
                dim=0,
            )
