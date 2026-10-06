from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str = "revenue_demand_forecasting",
    log_file: str | Path = "artifacts/app.log",
) -> logging.Logger:
    """Create one project logger for console and file output."""

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when this function is called again.
    if logger.handlers:
        return logger

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = get_logger()
