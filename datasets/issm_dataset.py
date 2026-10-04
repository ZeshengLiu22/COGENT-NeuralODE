"""ISSM dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch

from .anuga_dataset import volumes_to_edge_index
from .base_dataset import BaseTemporalGraphDataset, TrajectoryData


def _load_container(path: Path) -> dict[str, Any]:
    """Load a generic ISSM payload from ``.mat``, ``.npz``, or ``.pt``."""

    if path.suffix == ".mat":
        return scipy.io.loadmat(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dictionary payload in {path}, got {type(payload)!r}")
        return payload
    raise ValueError(f"Unsupported ISSM file type: {path.suffix}")


def _parse_rate_from_filename(path: Path) -> float | None:
    match = re.search(r"_r(\d{3})", path.name)
    if match is None:
        return None
    return float(match.group(1))


def _build_edge_attr(x: np.ndarray, y: np.ndarray, base: np.ndarray, surface: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Construct simple static edge features inspired by the legacy ISSM pipeline."""

    src = edge_index[0]
    dst = edge_index[1]
    dx = x[src] - x[dst]
    dy = y[src] - y[dst]
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6

    radial = np.exp(-(dist / 1000.0))
    base_slope = (base[src] - base[dst]) / dist
    surface_slope = (surface[src] - surface[dst]) / dist
    return np.stack([radial, base_slope, surface_slope], axis=1).astype(np.float32)


class ISSMDataset(BaseTemporalGraphDataset):
    """Paper-aligned ISSM inputs with no future floating-field information.

    Static channels are initial bed/base, surface, speed, and a binary mask
    (one means floating/ocean, i.e. ``floating[0] < 0``). Forcing contains
    initial-mask basal melt and queried-time SMB; state is vx, vy, thickness.
    Coordinates are used only to construct auxiliary edge data.
    """

    dataset_name = "issm"

    def _load_trajectory(self, path: Path) -> TrajectoryData:
        payload = _load_container(path)

        if "S" in payload:
            struct = payload["S"][0][0]
            xc = np.asarray(struct[0], dtype=np.float32).reshape(-1)
            yc = np.asarray(struct[1], dtype=np.float32).reshape(-1)
            elements = np.asarray(struct[2], dtype=np.int64) - 1
            smb = np.asarray(struct[3], dtype=np.float32)
            vx = np.asarray(struct[4], dtype=np.float32)
            vy = np.asarray(struct[5], dtype=np.float32)
            surface = np.asarray(struct[7], dtype=np.float32)
            base = np.asarray(struct[8], dtype=np.float32)
            thickness = np.asarray(struct[9], dtype=np.float32)
            floating = np.asarray(struct[10], dtype=np.float32)

            edge_index = volumes_to_edge_index(elements)
            edge_attr = _build_edge_attr(xc, yc, base[0], surface[0], edge_index)

            rate = _parse_rate_from_filename(path) or 0.0
            initial_mask = (floating[0] < 0.0).astype(np.float32)
            melt_rate = (initial_mask * rate).astype(np.float32)
            melt_force = np.broadcast_to(melt_rate[None, :, None], (thickness.shape[0], thickness.shape[1], 1)).copy()
            smb_force = smb[..., None].astype(np.float32)
            force = np.concatenate([melt_force, smb_force], axis=-1)

            state = np.stack([vx, vy, thickness], axis=-1).astype(np.float32)
            speed0 = np.sqrt(vx[0] ** 2 + vy[0] ** 2)
            x_static = np.stack([base[0], surface[0], speed0, initial_mask], axis=1).astype(np.float32)
            times = np.arange(state.shape[0], dtype=np.float32)
            scenario_param = np.asarray([[rate]], dtype=np.float32)
            scenario_id = path.stem
            sim_id = path.stem
        else:
            if "x_static" not in payload or "force" not in payload or "state" not in payload or "edge_index" not in payload:
                raise KeyError("Generic ISSM payloads require x_static, force, state, and edge_index.")
            x_static = np.asarray(payload["x_static"], dtype=np.float32)
            force = np.asarray(payload["force"], dtype=np.float32)
            state = np.asarray(payload["state"], dtype=np.float32)
            if x_static.shape[-1] != 4 or force.shape[-1] != 2 or state.shape[-1] != 3:
                raise ValueError(
                    "Generic ISSM payloads must follow the canonical protocol: "
                    "4 static channels [base0, surface0, speed0, floating/ocean mask], "
                    "2 forcing channels [basal melt, SMB], and 3 state channels [vx, vy, thickness]."
                )
            edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(payload["edge_attr"], dtype=np.float32) if "edge_attr" in payload else None
            times = np.asarray(payload.get("times", np.arange(state.shape[0])), dtype=np.float32).reshape(-1)
            scenario_param = None
            if "scenario_param" in payload:
                scenario_param = np.asarray(payload["scenario_param"], dtype=np.float32).reshape(1, -1)
            scenario_id = str(payload.get("scenario_id", path.stem))
            sim_id = str(payload.get("sim_id", path.stem))

        return TrajectoryData(
            x_static=x_static,
            force=force,
            state=state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            times=times,
            scenario_id=scenario_id,
            sim_id=sim_id,
            scenario_param=scenario_param,
        )
