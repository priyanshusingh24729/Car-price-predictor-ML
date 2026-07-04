"""
utils.py
========
Shared, dependency-light utility functions used across the project:
- Logger configuration (consistent format across all modules)
- Safe pickle save/load with clear error messages
- Directory creation helper

Keeping these here avoids duplicating boilerplate in
preprocessing.py, train.py, predict.py, and evaluate.py.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Logs to both console and a rotating project log file so that
    training/prediction runs are auditable after the fact.

    Args:
        name: Usually `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if the logger already exists
        # (can happen in interactive/Streamlit reruns).
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOGS_DIR / "project.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the filesystem is read-only (e.g. some deployment targets),
        # fall back silently to console-only logging.
        pass

    return logger


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    """
    Serialize any Python object to disk with pickle.

    Args:
        obj: Object to serialize (model, encoder, metadata dict, ...).
        path: Destination file path. Parent directories are created
            automatically.

    Raises:
        IOError: If the file cannot be written.
    """
    ensure_dir(path.parent)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except (OSError, pickle.PickleError) as exc:
        raise IOError(f"Failed to save object to {path}: {exc}") from exc


def load_pickle(path: Path) -> Any:
    """
    Load a pickled Python object from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file exists but cannot be unpickled.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found at {path}. "
            "Did you run `python main.py --stage train` first?"
        )
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError) as exc:
        raise IOError(f"Failed to load object from {path}: {exc}") from exc
