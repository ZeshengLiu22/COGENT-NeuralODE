"""Window enumeration and epoch-level subsampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WindowMetadata:
    """Metadata describing a temporal graph window."""

    scenario_index: int
    t_end: int


def enumerate_window_end_indices(
    total_steps: int,
    history_len: int,
    future_len: int,
    stride: int,
    window_reference: dict[str, int] | None = None,
) -> list[int]:
    """Return anchors valid for the requested and optional reference windows."""

    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if future_len < 1:
        raise ValueError("future_len must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    reference = window_reference or {}
    reference_history = int(reference.get("history_len", history_len))
    reference_future = int(reference.get("future_len", future_len))
    if reference_history < 1 or reference_future < 1:
        raise ValueError("window_reference history_len and future_len must be >= 1")
    min_t = max(history_len, reference_history) - 1
    max_t = total_steps - max(future_len, reference_future) - 1
    if max_t < min_t:
        return []
    return list(range(min_t, max_t + 1, stride))


def expand_windows(
    scenario_index: int,
    total_steps: int,
    history_len: int,
    future_len: int,
    stride: int,
    window_reference: dict[str, int] | None = None,
) -> list[WindowMetadata]:
    """Enumerate windows for a single scenario."""

    return [
        WindowMetadata(scenario_index=scenario_index, t_end=t_end)
        for t_end in enumerate_window_end_indices(total_steps, history_len, future_len, stride, window_reference)
    ]


def sample_windows_for_epoch(
    windows: Iterable[WindowMetadata],
    scenario_count: int,
    rng: np.random.Generator,
    windows_per_scenario: int | None = None,
    epoch_num_windows: int | None = None,
) -> list[WindowMetadata]:
    """Sample an epoch subset while preserving scenario-first semantics."""

    windows = list(windows)
    if windows_per_scenario is None and epoch_num_windows is None:
        return windows

    scenario_groups: dict[int, list[WindowMetadata]] = {idx: [] for idx in range(scenario_count)}
    for window in windows:
        scenario_groups[window.scenario_index].append(window)

    sampled: list[WindowMetadata] = []
    for scenario_index in range(scenario_count):
        group = scenario_groups[scenario_index]
        if not group:
            continue

        if windows_per_scenario is None or windows_per_scenario >= len(group):
            sampled.extend(group)
            continue

        choice = rng.choice(len(group), size=windows_per_scenario, replace=False)
        sampled.extend(group[int(idx)] for idx in choice)

    if epoch_num_windows is not None and len(sampled) > epoch_num_windows:
        choice = rng.choice(len(sampled), size=epoch_num_windows, replace=False)
        sampled = [sampled[int(idx)] for idx in choice]

    return sampled
