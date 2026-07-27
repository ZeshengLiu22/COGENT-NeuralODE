"""ADCIRC dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch

from .anuga_dataset import volumes_to_edge_index
from .base_dataset import BaseTemporalGraphDataset, TrajectoryData


def _load_container(path: Path) -> dict[str, Any]:
    """Load a generic ADCIRC container from ``.npz``, ``.pt``, or ``.mat``."""

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dictionary payload in {path}, got {type(payload)!r}")
        return payload
    if path.suffix == ".mat":
        return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}
    raise ValueError(f"Unsupported ADCIRC file type: {path.suffix}")


class ADCIRCDataset(BaseTemporalGraphDataset):
    """Loader for ADCIRC storm-surge trajectories."""

    dataset_name = "adcirc"

    def _load_trajectory(self, path: Path) -> TrajectoryData:
        payload = _load_container(path)

        if "x_static" in payload:
            x_static = np.asarray(payload["x_static"], dtype=np.float32)
        elif "coords" in payload:
            x_static = np.asarray(payload["coords"], dtype=np.float32)
        elif "x" in payload and "y" in payload:
            x_static = np.stack([np.asarray(payload["x"], dtype=np.float32), np.asarray(payload["y"], dtype=np.float32)], axis=1)
        else:
            raise KeyError("ADCIRC data requires x_static, coords, or x/y arrays.")

        if "force" in payload:
            force = np.asarray(payload["force"], dtype=np.float32)
        else:
            required_force = ["wind_vx", "wind_vy", "pressure"]
            if not all(key in payload for key in required_force):
                raise KeyError("ADCIRC data requires force or wind_vx/wind_vy/pressure arrays.")
            force = np.stack(
                [
                    np.asarray(payload["wind_vx"], dtype=np.float32),
                    np.asarray(payload["wind_vy"], dtype=np.float32),
                    np.asarray(payload["pressure"], dtype=np.float32),
                ],
                axis=-1,
            )

        if "state" in payload:
            state = np.asarray(payload["state"], dtype=np.float32)
        elif "surge" in payload:
            state = np.asarray(payload["surge"], dtype=np.float32)[..., None]
        else:
            raise KeyError("ADCIRC data requires state or surge arrays.")

        if "times" in payload:
            times = np.asarray(payload["times"], dtype=np.float32).reshape(-1)
        elif "time" in payload:
            times = np.asarray(payload["time"], dtype=np.float32).reshape(-1)
        else:
            times = np.arange(state.shape[0], dtype=np.float32)

        if "edge_index" in payload:
            edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
        elif "elements" in payload:
            edge_index = volumes_to_edge_index(np.asarray(payload["elements"], dtype=np.int64) - int(np.min(payload["elements"])))
        elif "triangles" in payload:
            edge_index = volumes_to_edge_index(np.asarray(payload["triangles"], dtype=np.int64) - int(np.min(payload["triangles"])))
        else:
            raise KeyError("ADCIRC data requires edge_index, elements, or triangles.")

        edge_attr = None
        if "edge_attr" in payload:
            edge_attr = np.asarray(payload["edge_attr"], dtype=np.float32)

        scenario_id = str(payload.get("scenario_id", path.stem))
        sim_id = str(payload.get("sim_id", path.stem))
        scenario_param = None
        if "scenario_param" in payload:
            scenario_param = np.asarray(payload["scenario_param"], dtype=np.float32).reshape(1, -1)

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
