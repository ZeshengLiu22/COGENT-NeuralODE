"""Natural training-anchor enumeration and per-scenario series sampling."""

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
) -> list[int]:
    """Return all anchors with H historical states and K future states."""

    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if future_len < 1:
        raise ValueError("future_len must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    min_t = history_len - 1
    max_t = total_steps - future_len - 1
    if max_t < min_t:
        return []
    return list(range(min_t, max_t + 1, stride))


def expand_windows(
    scenario_index: int,
    total_steps: int,
    history_len: int,
    future_len: int,
    stride: int,
) -> list[WindowMetadata]:
    """Enumerate windows for a single scenario."""

    return [
        WindowMetadata(scenario_index=scenario_index, t_end=t_end)
        for t_end in enumerate_window_end_indices(total_steps, history_len, future_len, stride)
    ]


def sample_training_series_for_epoch(
    natural_series: Iterable[WindowMetadata],
    scenario_count: int,
    rng: np.random.Generator,
    train_series_per_scenario_per_epoch: int | None = None,
) -> list[WindowMetadata]:
    """Select the configured number of training series for every scenario.

    A series is identified by a scenario and natural history anchor. Its actual
    prediction length is determined later by the independently sampled k_eff.
    Include every unique anchor before drawing replacement extras when a
    scenario has fewer natural anchors than the budget.
    """

    natural_series = list(natural_series)
    budget = train_series_per_scenario_per_epoch
    if budget is None:
        return natural_series
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("train_series_per_scenario_per_epoch must be a positive integer or null.")

    scenario_groups: dict[int, list[WindowMetadata]] = {idx: [] for idx in range(scenario_count)}
    for series in natural_series:
        scenario_groups[series.scenario_index].append(series)

    sampled: list[WindowMetadata] = []
    for scenario_index in range(scenario_count):
        group = scenario_groups[scenario_index]
        if not group:
            raise ValueError(
                f"Training scenario {scenario_index} has no natural anchors for its H/K; "
                "cannot provide train_series_per_scenario_per_epoch."
            )
        if len(group) >= budget:
            choice = rng.choice(len(group), size=budget, replace=False)
        else:
            extras = rng.choice(len(group), size=budget - len(group), replace=True)
            choice = np.concatenate((np.arange(len(group)), extras))
            rng.shuffle(choice)
        sampled.extend(group[int(index)] for index in choice)

    return sampled
