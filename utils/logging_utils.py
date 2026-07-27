"""Lightweight logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .seed import get_rank


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a simple process-aware logger."""

    logger = logging.getLogger("continuous_graph_emulator")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(Path(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def rank0_log(logger: logging.Logger, message: str) -> None:
    """Log from rank zero only."""

    if get_rank() == 0:
        logger.info(message)
