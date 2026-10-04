"""Config and filesystem helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return the resolved path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config_bundle(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Load and recursively merge multiple YAML config files."""

    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_update(merged, load_yaml(path))
    return merged


def save_json(path: str | Path, payload: Any) -> None:
    """Write a JSON payload with stable formatting."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a PyTorch checkpoint atomically enough for normal training usage."""

    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
