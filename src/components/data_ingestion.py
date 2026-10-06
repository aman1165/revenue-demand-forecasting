from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataIngestion:
    """Load every raw CSV used by the forecasting project."""

    def __init__(self, raw_data_dir: str | Path):
        self.raw_data_dir = Path(raw_data_dir)

    def _read_csv(
        self,
        file_name: str,
        date_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read one CSV and give a useful error when it is missing."""
        file_path = self.raw_data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

        return pd.read_csv(
            file_path,
            parse_dates=date_columns,
        )

    def load_train(self) -> pd.DataFrame:
        return self._read_csv(
            "train.csv",
            date_columns=["date"],
        )

    def load_test(self) -> pd.DataFrame:
        return self._read_csv(
            "test.csv",
            date_columns=["date"],
        )

    def load_stores(self) -> pd.DataFrame:
        return self._read_csv("stores.csv")

    def load_oil(self) -> pd.DataFrame:
        return self._read_csv(
            "oil.csv",
            date_columns=["date"],
        )

    def load_holidays(self) -> pd.DataFrame:
        return self._read_csv(
            "holidays_events.csv",
            date_columns=["date"],
        )

    def load_transactions(self) -> pd.DataFrame:
        return self._read_csv(
            "transactions.csv",
            date_columns=["date"],
        )

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all project datasets in one place."""
        return {
            "train": self.load_train(),
            "test": self.load_test(),
            "stores": self.load_stores(),
            "oil": self.load_oil(),
            "holidays": self.load_holidays(),
            "transactions": self.load_transactions(),
        }
