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


def validate_split_files(split_files: dict[str, Sequence[str | Path]]) -> None:
    """Reject duplicate scenarios within or across the saved splits."""

    seen: dict[Path, str] = {}
    for split in ("train", "val", "test"):
        if split not in split_files:
            raise ValueError(f"Split manifest is missing {split!r}.")
        for value in split_files[split]:
            path = Path(value).resolve()
            if path in seen:
                raise ValueError(f"Scenario {value!s} occurs in both {seen[path]} and {split} split entries.")
            seen[path] = split


def make_split_manifest(split_files: dict[str, Sequence[str | Path]], data_dir: str | Path) -> dict[str, list[str]]:
    """Record stable scenario identifiers relative to the dataset directory."""

    validate_split_files(split_files)
    root = Path(data_dir).resolve()
    manifest: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        manifest[split] = []
        for value in split_files[split]:
            path = Path(value).resolve()
            try:
                identifier = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Scenario {path} is outside dataset.data_dir={root}.") from exc
            manifest[split].append(identifier)
    return manifest


def resolve_split_manifest(manifest: dict[str, Sequence[str]], data_dir: str | Path) -> dict[str, list[Path]]:
    """Restore only the recorded files, including after relocating the data root."""

    root = Path(data_dir).resolve()
    split_files: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        if split not in manifest:
            raise ValueError(f"Split manifest is missing {split!r}.")
        split_files[split] = []
        for identifier in manifest[split]:
            relative = Path(identifier)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Split identifier must be a path relative to data_dir: {identifier!r}.")
            path = (root / relative).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Saved {split} scenario is missing: {path}")
            split_files[split].append(path)
    validate_split_files(split_files)
    return split_files
