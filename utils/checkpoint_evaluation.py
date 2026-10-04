"""Restore evaluation provenance and apply a small, explicit runtime whitelist."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from datasets.split_utils import resolve_split_manifest


_RUNTIME_PATHS = {
    ("dataset", "data_dir"),
    ("dataset", "history_len"),
    ("dataset", "future_len"),
    ("evaluation", "batch_size"),
    ("evaluation", "num_workers"),
    ("evaluation", "amp_mode"),
    ("output_dir",),
    ("device",),
}


def restore_evaluation_config(checkpoint: dict, external_config: dict | None = None, *, runtime_overrides: dict | None = None) -> dict:
    """Use the checkpoint model/config; reject changes beyond evaluation runtime."""

    if "config" not in checkpoint:
        raise ValueError("Checkpoint has no saved training config; evaluation cannot reconstruct its model.")
    saved = checkpoint["config"]
    config = deepcopy(saved)
    config.setdefault("evaluation", {})["amp_mode"] = "none"

    def apply(override: dict, target: dict, original: dict, path: tuple[str, ...] = ()) -> None:
        for key, value in override.items():
            full_path = path + (key,)
            if full_path in _RUNTIME_PATHS:
                target[key] = deepcopy(value)
            elif isinstance(value, dict) and (
                isinstance(original.get(key), dict)
                or any(allowed[:len(full_path)] == full_path for allowed in _RUNTIME_PATHS)
            ):
                apply(value, target.setdefault(key, {}), original.get(key, {}), full_path)
            elif key not in original or value != original[key]:
                name = ".".join(full_path)
                raise ValueError(f"Evaluation cannot override {name}; the checkpoint training config is authoritative.")

    for override in (external_config, runtime_overrides):
        if override:
            apply(override, config, saved)
    for name in ("history_len", "future_len"):
        if int(config["dataset"][name]) < 1:
            raise ValueError(f"Evaluation {name} must be >= 1.")
    if int(config["evaluation"].get("batch_size", 1)) < 1:
        raise ValueError("Evaluation batch_size must be >= 1.")
    if int(config["evaluation"].get("num_workers", 0)) < 0:
        raise ValueError("Evaluation num_workers must be >= 0.")
    if config["evaluation"]["amp_mode"] not in ("none", "bf16", "fp16"):
        raise ValueError("Evaluation amp_mode must be none, bf16, or fp16.")
    return config


def restore_checkpoint_splits(checkpoint: dict, data_dir: str | Path) -> dict[str, list[Path]]:
    """Restore the embedded split without discovering or resplitting new files."""

    if "split_manifest" not in checkpoint:
        raise ValueError("Checkpoint has no split_manifest; exact evaluation requires a checkpoint with its saved split.")
    return resolve_split_manifest(checkpoint["split_manifest"], data_dir)


def evaluation_autocast(device: torch.device, amp_mode: str = "none"):
    """Keep standalone inference FP32 unless evaluation AMP is requested."""

    if device.type != "cuda" or amp_mode == "none":
        return nullcontext()
    if amp_mode not in ("bf16", "fp16"):
        raise ValueError(f"Unknown evaluation AMP mode: {amp_mode}")
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp_mode == "bf16" else torch.float16)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation, but no CUDA device is available.")
    return device
