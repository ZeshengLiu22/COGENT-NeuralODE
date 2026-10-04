"""Shared temporal graph dataset logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .normalization import FeatureNormalizer
from .window_utils import WindowMetadata, expand_windows, sample_windows_for_epoch


def _require_finite(name: str, array: np.ndarray) -> None:
    """Raise a helpful error if an array contains NaN or inf values."""

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")


def _normalized_relative_times(times: np.ndarray, anchor_index: int, index_slice: slice) -> torch.Tensor:
    """Normalize relative times so continuous solvers operate on an O(1) horizon."""

    offsets = np.asarray(times[index_slice], dtype=np.float32) - np.float32(times[anchor_index])
    deltas = np.diff(np.asarray(times, dtype=np.float64))
    positive_deltas = deltas[deltas > 0.0]
    time_scale = float(np.median(positive_deltas)) if positive_deltas.size else 1.0
    if not np.isfinite(time_scale) or time_scale <= 0.0:
        time_scale = 1.0
    return torch.as_tensor(offsets / np.float32(time_scale), dtype=torch.float32).unsqueeze(0)


@dataclass
class TrajectoryData:
    """Unified trajectory representation used by all dataset adapters."""

    x_static: np.ndarray
    force: np.ndarray
    state: np.ndarray
    edge_index: np.ndarray
    times: np.ndarray
    scenario_id: str
    sim_id: str
    edge_attr: Optional[np.ndarray] = None
    scenario_param: Optional[np.ndarray] = None
    target_mask: Optional[np.ndarray] = None

    def validate(self) -> "TrajectoryData":
        """Validate and normalize dtypes for downstream code."""

        self.x_static = np.asarray(self.x_static, dtype=np.float32)
        self.force = np.asarray(self.force, dtype=np.float32)
        self.state = np.asarray(self.state, dtype=np.float32)
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        self.times = np.asarray(self.times, dtype=np.float32)

        _require_finite("x_static", self.x_static)
        _require_finite("force", self.force)
        _require_finite("state", self.state)
        _require_finite("times", self.times)

        if self.x_static.ndim != 2:
            raise ValueError(f"x_static must have shape [N, F_static], got {self.x_static.shape}")
        if self.state.ndim != 3:
            raise ValueError(f"state must have shape [T, N, F_state], got {self.state.shape}")
        if self.force.ndim == 2:
            self.force = np.broadcast_to(self.force[:, None, :], (self.force.shape[0], self.x_static.shape[0], self.force.shape[1])).copy()
        if self.force.ndim != 3:
            raise ValueError(f"force must have shape [T, N, F_force] or [T, F_force], got {self.force.shape}")
        if self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {self.edge_index.shape}")
        if self.state.shape[0] != self.force.shape[0]:
            raise ValueError("force and state must share the same trajectory length.")
        if self.state.shape[1] != self.x_static.shape[0] or self.force.shape[1] != self.x_static.shape[0]:
            raise ValueError("Static, forcing, and state arrays must share the same node count.")
        if self.times.ndim != 1 or self.times.shape[0] != self.state.shape[0]:
            raise ValueError("times must have shape [T] and match the trajectory length.")
        if np.any(np.diff(self.times.astype(np.float64)) <= 0.0):
            raise ValueError("times must be strictly increasing for continuous-time interpolation.")
        if self.edge_attr is not None:
            self.edge_attr = np.asarray(self.edge_attr, dtype=np.float32)
            _require_finite("edge_attr", self.edge_attr)
        if self.scenario_param is not None:
            self.scenario_param = np.asarray(self.scenario_param, dtype=np.float32).reshape(1, -1)
            _require_finite("scenario_param", self.scenario_param)
        if self.target_mask is not None:
            self.target_mask = np.asarray(self.target_mask, dtype=np.float32)
            _require_finite("target_mask", self.target_mask)
        return self


class BaseTemporalGraphDataset(Dataset):
    """Scenario-first temporal graph dataset with fixed history and future windows."""

    dataset_name = "base"

    def __init__(
        self,
        scenario_files: Sequence[str | Path],
        history_len: int,
        future_len: int,
        split: str,
        stride: int = 1,
        normalizer: FeatureNormalizer | None = None,
        cache_in_memory: bool = False,
        epoch_num_windows: int | None = None,
        windows_per_scenario: int | None = None,
        seed: int = 42,
        adapter_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario_files = [Path(path) for path in scenario_files]
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.split = split
        self.stride = int(stride)
        self.normalizer = normalizer
        self.cache_in_memory = bool(cache_in_memory)
        self.epoch_num_windows = epoch_num_windows
        self.windows_per_scenario = windows_per_scenario
        self.seed = int(seed)
        self.adapter_kwargs = adapter_kwargs or {}

        self._trajectory_cache: dict[int, TrajectoryData] = {}
        self.scenario_infos: list[dict[str, Any]] = []
        self.all_windows: list[WindowMetadata] = []

        raw_ids: list[str] = []
        sim_ids: list[str] = []

        for scenario_index in range(len(self.scenario_files)):
            trajectory = self._get_trajectory(scenario_index, cache=self.cache_in_memory)
            raw_ids.append(trajectory.scenario_id)
            sim_ids.append(trajectory.sim_id)
            self.scenario_infos.append(
                {
                    "path": self.scenario_files[scenario_index],
                    "scenario_id": trajectory.scenario_id,
                    "sim_id": trajectory.sim_id,
                    "length": int(trajectory.times.shape[0]),
                }
            )
            self.all_windows.extend(
                expand_windows(
                    scenario_index=scenario_index,
                    total_steps=trajectory.times.shape[0],
                    history_len=self.history_len,
                    future_len=self.future_len,
                    stride=self.stride,
                )
            )

        self.scenario_id_to_code = {name: idx for idx, name in enumerate(sorted(set(raw_ids)))}
        self.sim_id_to_code = {name: idx for idx, name in enumerate(sorted(set(sim_ids)))}
        self.active_windows: list[WindowMetadata] = list(self.all_windows)
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.active_windows)

    def __getitem__(self, index: int) -> Data:
        window = self.active_windows[index]
        trajectory = self._get_trajectory(window.scenario_index, cache=self.cache_in_memory)
        t_end = window.t_end

        hist_slice = slice(t_end - self.history_len + 1, t_end + 1)
        future_slice = slice(t_end + 1, t_end + 1 + self.future_len)

        x_static = torch.as_tensor(trajectory.x_static, dtype=torch.float32)
        state_hist = torch.as_tensor(trajectory.state[hist_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        force_hist = torch.as_tensor(trajectory.force[hist_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        force_future = torch.as_tensor(trajectory.force[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        y_future = torch.as_tensor(trajectory.state[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()

        if self.normalizer is not None:
            x_static = self.normalizer.transform_static(x_static)
            state_hist = self.normalizer.transform_state(state_hist)
            force_hist = self.normalizer.transform_force(force_hist)
            force_future = self.normalizer.transform_force(force_future)
            y_future = self.normalizer.transform_state(y_future)

        t_hist = _normalized_relative_times(trajectory.times, t_end, hist_slice)
        t_future = _normalized_relative_times(trajectory.times, t_end, future_slice)

        data = Data(
            x_static=x_static,
            state_hist=state_hist,
            force_hist=force_hist,
            force_future=force_future,
            y_future=y_future,
            edge_index=torch.as_tensor(trajectory.edge_index, dtype=torch.long),
        )
        data.num_nodes = int(x_static.shape[0])
        data.t_hist = t_hist
        data.t_future = t_future
        data.future_idx = torch.arange(t_end + 1, t_end + 1 + self.future_len, dtype=torch.long).unsqueeze(0)
        data.future_time = torch.as_tensor(trajectory.times[future_slice], dtype=torch.float32).unsqueeze(0)
        data.scenario_id = torch.tensor([self.scenario_id_to_code[trajectory.scenario_id]], dtype=torch.long)
        data.sim_id = torch.tensor([self.sim_id_to_code[trajectory.sim_id]], dtype=torch.long)
        data.scenario_idx = torch.tensor([window.scenario_index], dtype=torch.long)
        data.t_idx = torch.tensor([t_end], dtype=torch.long)
        data.t_value = torch.tensor([float(trajectory.times[t_end])], dtype=torch.float32)

        if trajectory.edge_attr is not None:
            data.edge_attr = torch.as_tensor(trajectory.edge_attr, dtype=torch.float32)
        if trajectory.scenario_param is not None:
            data.scenario_param = torch.as_tensor(trajectory.scenario_param, dtype=torch.float32)
        if trajectory.target_mask is not None:
            target_mask = torch.as_tensor(trajectory.target_mask[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
            data.target_mask = target_mask
        return data

    def set_epoch(self, epoch: int) -> None:
        """Resample active windows for the current epoch when configured."""

        if self.split != "train":
            self.active_windows = list(self.all_windows)
            return

        rng = np.random.default_rng(self.seed + int(epoch))
        self.active_windows = sample_windows_for_epoch(
            windows=self.all_windows,
            scenario_count=len(self.scenario_files),
            rng=rng,
            windows_per_scenario=self.windows_per_scenario,
            epoch_num_windows=self.epoch_num_windows,
        )

    def iter_trajectories(self) -> list[TrajectoryData]:
        """Return all trajectories in the dataset."""

        return [self._get_trajectory(index, cache=self.cache_in_memory) for index in range(len(self.scenario_files))]

    def get_rollout_data(self, scenario_index: int, start_t: int | None = None) -> Data:
        """Use H true states ending at ``start_t``, then predict to trajectory end.

        The formal rollout start is ``known_steps = start_t + 1``. Its history
        is ``[known_steps - H, known_steps)`` and its future is
        ``[known_steps, T)``; training ``future_len`` does not limit the future.
        """

        trajectory = self._get_trajectory(scenario_index, cache=self.cache_in_memory)
        total_steps = trajectory.times.shape[0]
        if start_t is None:
            start_t = self.history_len - 1
        if start_t < self.history_len - 1:
            raise ValueError("known_steps (start_t + 1) must be >= history_len.")
        if start_t >= total_steps - 1:
            raise ValueError("known_steps (start_t + 1) must be < trajectory length.")

        dynamic_future_len = total_steps - start_t - 1
        hist_slice = slice(start_t - self.history_len + 1, start_t + 1)
        future_slice = slice(start_t + 1, total_steps)

        x_static = torch.as_tensor(trajectory.x_static, dtype=torch.float32)
        state_hist = torch.as_tensor(trajectory.state[hist_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        force_hist = torch.as_tensor(trajectory.force[hist_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        force_future = torch.as_tensor(trajectory.force[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
        y_future = torch.as_tensor(trajectory.state[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()

        if self.normalizer is not None:
            x_static = self.normalizer.transform_static(x_static)
            state_hist = self.normalizer.transform_state(state_hist)
            force_hist = self.normalizer.transform_force(force_hist)
            force_future = self.normalizer.transform_force(force_future)
            y_future = self.normalizer.transform_state(y_future)

        data = Data(
            x_static=x_static,
            state_hist=state_hist,
            force_hist=force_hist,
            force_future=force_future,
            y_future=y_future,
            edge_index=torch.as_tensor(trajectory.edge_index, dtype=torch.long),
        )
        data.num_nodes = int(x_static.shape[0])
        data.t_hist = _normalized_relative_times(trajectory.times, start_t, hist_slice)
        data.t_future = _normalized_relative_times(trajectory.times, start_t, future_slice)
        data.history_idx = torch.arange(start_t - self.history_len + 1, start_t + 1, dtype=torch.long).unsqueeze(0)
        data.history_time = torch.as_tensor(trajectory.times[hist_slice], dtype=torch.float32).unsqueeze(0)
        data.future_idx = torch.arange(start_t + 1, total_steps, dtype=torch.long).unsqueeze(0)
        data.future_time = torch.as_tensor(trajectory.times[future_slice], dtype=torch.float32).unsqueeze(0)
        data.scenario_id = torch.tensor([self.scenario_id_to_code[trajectory.scenario_id]], dtype=torch.long)
        data.sim_id = torch.tensor([self.sim_id_to_code[trajectory.sim_id]], dtype=torch.long)
        data.scenario_idx = torch.tensor([scenario_index], dtype=torch.long)
        data.t_idx = torch.tensor([start_t], dtype=torch.long)
        data.t_value = torch.tensor([float(trajectory.times[start_t])], dtype=torch.float32)
        data.rollout_length = torch.tensor([dynamic_future_len], dtype=torch.long)
        if trajectory.edge_attr is not None:
            data.edge_attr = torch.as_tensor(trajectory.edge_attr, dtype=torch.float32)
        if trajectory.scenario_param is not None:
            data.scenario_param = torch.as_tensor(trajectory.scenario_param, dtype=torch.float32)
        if trajectory.target_mask is not None:
            target_mask = torch.as_tensor(trajectory.target_mask[future_slice], dtype=torch.float32).permute(1, 0, 2).contiguous()
            data.target_mask = target_mask
        return data

    def _get_trajectory(self, scenario_index: int, cache: bool) -> TrajectoryData:
        if scenario_index in self._trajectory_cache:
            return self._trajectory_cache[scenario_index]

        trajectory = self._load_trajectory(self.scenario_files[scenario_index]).validate()
        if cache:
            self._trajectory_cache[scenario_index] = trajectory
        return trajectory

    def _load_trajectory(self, path: Path) -> TrajectoryData:
        raise NotImplementedError
