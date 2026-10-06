from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import yaml


def ensure_directory(path: str | Path) -> Path:
    """Create a directory when it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(config_path: str | Path) -> dict:
    """Load the YAML project configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def save_object(obj: Any, file_path: str | Path) -> Path:
    """Save any Python object with joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, file_path)

    return file_path


def load_object(file_path: str | Path) -> Any:
    """Load a joblib object."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Object file not found: {file_path}"
        )

    return joblib.load(file_path)


def save_dataframe(dataframe, file_path: str | Path) -> Path:
    """Save a pandas DataFrame as CSV."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(file_path, index=False)

    return file_path
