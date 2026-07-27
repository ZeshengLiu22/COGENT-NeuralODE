"""Scenario-discovery and split helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def discover_files(data_dir: str | Path, patterns: str | Sequence[str]) -> list[Path]:
    """Discover scenario files using one or more glob patterns."""

    root = Path(data_dir)
    pattern_list = [patterns] if isinstance(patterns, str) else list(patterns)
    found: list[Path] = []
    for pattern in pattern_list:
        found.extend(sorted(root.glob(pattern)))
    unique = sorted(set(found))
    if not unique:
        raise FileNotFoundError(f"No files found under {root} with patterns={pattern_list}")
    return unique


def random_split(paths: Sequence[Path], train: float, val: float, test: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a file list by shuffled scenario."""

    total = train + val + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train + val + test must sum to 1.0")

    paths = list(paths)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(n_total * train))
    n_val = int(round(n_total * val))

    train_paths = [paths[int(idx)] for idx in indices[:n_train]]
    val_paths = [paths[int(idx)] for idx in indices[n_train : n_train + n_val]]
    test_paths = [paths[int(idx)] for idx in indices[n_train + n_val :]]
    return train_paths, val_paths, test_paths


def issm_rate_modulo_split(paths: Iterable[Path], modulo: int = 20, val_remainder: int = 0, test_remainder: int = 10) -> tuple[list[Path], list[Path], list[Path]]:
    """Split ISSM files by melt-rate modulo, following the legacy convention."""

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    test_paths: list[Path] = []

    for path in sorted(paths):
        rate = parse_issm_rate_from_filename(path)
        if rate is None:
            raise ValueError(f"Could not parse ISSM melt rate from filename: {path}")
        remainder = rate % modulo
        if remainder == val_remainder:
            val_paths.append(Path(path))
        elif remainder == test_remainder:
            test_paths.append(Path(path))
        else:
            train_paths.append(Path(path))
    return train_paths, val_paths, test_paths


def parse_issm_rate_from_filename(path: str | Path) -> int | None:
    """Extract a three-digit melt-rate tag from an ISSM filename."""

    match = re.search(r"_r(\d{3})", str(path))
    if match is None:
        return None
    return int(match.group(1))
