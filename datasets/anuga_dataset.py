"""ANUGA dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base_dataset import BaseTemporalGraphDataset, TrajectoryData


def volumes_to_edge_index(volumes: np.ndarray) -> np.ndarray:
    """Convert triangular cell connectivity into a deduplicated undirected edge index."""

    volumes = np.asarray(volumes, dtype=np.int64)
    if volumes.ndim != 2 or volumes.shape[1] != 3:
        raise ValueError(f"Expected triangle volumes with shape [M, 3], got {volumes.shape}")

    edges = np.concatenate(
        [
            volumes[:, [0, 1]],
            volumes[:, [1, 2]],
            volumes[:, [2, 0]],
        ],
        axis=0,
    )
    a = np.minimum(edges[:, 0], edges[:, 1])
    b = np.maximum(edges[:, 0], edges[:, 1])
    undirected = np.unique(np.stack([a, b], axis=1), axis=0)
    src = np.concatenate([undirected[:, 0], undirected[:, 1]], axis=0)
    dst = np.concatenate([undirected[:, 1], undirected[:, 0]], axis=0)
    return np.stack([src, dst], axis=0).astype(np.int64)


def compute_depth(stage: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Compute nonnegative water depth from stage and elevation."""

    return np.maximum(stage - elevation[None, :], 0.0).astype(np.float32)


class ANUGADataset(BaseTemporalGraphDataset):
    """Loader for merged ANUGA flood trajectories stored as ``.npz`` files."""

    dataset_name = "anuga"

    def _load_trajectory(self, path: Path) -> TrajectoryData:
        data = np.load(path)

        x_coord = np.asarray(data["x"], dtype=np.float32)
        y_coord = np.asarray(data["y"], dtype=np.float32)
        elevation = np.asarray(data["elevation"], dtype=np.float32)
        friction = np.asarray(data["friction"], dtype=np.float32)
        x_static = np.stack([x_coord, y_coord, elevation, friction], axis=1)

        if "edge_index" in data:
            edge_index = np.asarray(data["edge_index"], dtype=np.int64)
        elif "volumes" in data:
            edge_index = volumes_to_edge_index(np.asarray(data["volumes"], dtype=np.int64))
        else:
            raise KeyError("ANUGA files must contain either edge_index or volumes.")

        times = np.asarray(data["time"], dtype=np.float32)
        rain_rate = np.asarray(data["rain_rate"], dtype=np.float32)
        force = np.broadcast_to(rain_rate[:, None, None], (times.shape[0], x_static.shape[0], 1)).copy()

        stage = np.asarray(data["stage"], dtype=np.float32)
        xmomentum = np.asarray(data["xmomentum"], dtype=np.float32)
        ymomentum = np.asarray(data["ymomentum"], dtype=np.float32)
        if "h" in data:
            depth = np.asarray(data["h"], dtype=np.float32)
        else:
            depth = compute_depth(stage, elevation)
        state = np.stack([depth, xmomentum, ymomentum], axis=-1)

        scenario_param = None
        if "scenario_param" in data:
            scenario_param = np.asarray(data["scenario_param"], dtype=np.float32).reshape(1, -1)
        else:
            scenario_param = np.asarray([[float(np.max(rain_rate))]], dtype=np.float32)

        return TrajectoryData(
            x_static=x_static,
            force=force,
            state=state,
            edge_index=edge_index,
            times=times,
            scenario_id=path.stem.replace("_merged", ""),
            sim_id=path.stem,
            scenario_param=scenario_param,
        )
