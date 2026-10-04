"""Spatial and temporal history encoder for NODE2."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.common.gnn_blocks import GraphNetwork
from models.common.lstm_blocks import NodeWiseLSTM
from models.common.mlp import MLP, get_activation


class HistoryEncoder(nn.Module):
    """Static MLP + per-step GNN + temporal history summarizer.

    The public tensor contract stays node-major for PyG batches:
    ``step_embeds`` is ``[N_total, H, D_enc]`` and ``hist_context`` is
    ``[N_total, D_hist]``.  The default temporal summarizer is a
    temporal-only Transformer over the history dimension for each node
    independently; a node-wise LSTM is also available.
    """

    def __init__(self, num_static: int, num_force: int, num_state: int, config: dict) -> None:
        super().__init__()
        history_cfg = config["model"]["history_encoder"]
        static_hidden = history_cfg.get("static_hidden_dims", [64])
        static_embed_dim = history_cfg["static_embed_dim"]
        gnn_hidden_dim = history_cfg["gnn_hidden_dim"]
        gnn_layers = history_cfg.get("gnn_num_layers", 2)
        gnn_type = history_cfg.get("gnn_type", "sage")
        gnn_activation = history_cfg.get("gnn_activation", "gelu")
        gnn_dropout = history_cfg.get("dropout", 0.0)
        lstm_hidden_dim = history_cfg["lstm_hidden_dim"]
        lstm_layers = history_cfg.get("lstm_num_layers", 1)
        self.history_encoder_type = history_cfg.get("history_encoder_type", "transformer")
        self.history_context_pooling = history_cfg.get("history_context_pooling", "last")
        self.history_use_positional_encoding = history_cfg.get("history_use_positional_encoding", True)
        self.history_transformer_chunk_size = history_cfg.get("history_transformer_chunk_size", 8192)
        self.history_transformer_force_math_sdp = history_cfg.get("history_transformer_force_math_sdp", True)

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
        if self.history_encoder_type == "transformer":
            transformer_layers = history_cfg.get("history_transformer_num_layers", 2)
            transformer_heads = history_cfg.get("history_transformer_num_heads", 4)
            transformer_ff_dim = history_cfg.get("history_transformer_ff_dim", static_embed_dim * 4)
            transformer_dropout = history_cfg.get("history_transformer_dropout", gnn_dropout)
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
        elif self.history_encoder_type == "lstm":
            self.temporal_lstm = NodeWiseLSTM(
                static_embed_dim,
                lstm_hidden_dim,
                num_layers=lstm_layers,
                dropout=gnn_dropout,
            )
        else:
            raise ValueError(f"Unknown history encoder: {self.history_encoder_type!r}")

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
        static_embed = self.static_mlp(x_static)
        history_len = state_hist.shape[1]
        step_embeds: list[torch.Tensor] = []

        for step in range(history_len):
            step_inputs = torch.cat([state_hist[:, step, :], force_hist[:, step, :], static_embed], dim=-1)
            step_embeds.append(self.step_gnn(step_inputs, edge_index))

        # Node-major PyG batching: temporal modules see each node as one
        # independent sequence of H per-step spatial embeddings.
        stacked = torch.stack(step_embeds, dim=1)
        if self.history_encoder_type == "transformer":
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
            elif self.history_context_pooling == "last":
                pooled_context = temporal_outputs[:, -1, :]
            else:
                raise ValueError(f"Unknown history pooling: {self.history_context_pooling!r}")
            hist_context = self.transformer_context_projection(pooled_context)
        else:
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
        chunk_size = self.history_transformer_chunk_size
        force_math_sdp = temporal_inputs.is_cuda and self.history_transformer_force_math_sdp
        with sdpa_kernel(SDPBackend.MATH) if force_math_sdp else nullcontext():
            if chunk_size is None or temporal_inputs.shape[0] <= chunk_size:
                return self.temporal_transformer(temporal_inputs)
            return torch.cat(
                [self.temporal_transformer(chunk) for chunk in temporal_inputs.split(chunk_size, dim=0)],
                dim=0,
            )
