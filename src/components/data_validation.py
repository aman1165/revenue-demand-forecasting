from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataValidation:
    """Validate file availability and basic dataset quality."""

    REQUIRED_FILES = (
        "train.csv",
        "test.csv",
        "stores.csv",
        "oil.csv",
        "holidays_events.csv",
        "transactions.csv",
    )

    def __init__(self, raw_data_dir: str | Path):
        self.raw_data_dir = Path(raw_data_dir)

    def validate_files_exist(self) -> None:
        """Make sure every raw input file is available."""
        missing_files = [
            name
            for name in self.REQUIRED_FILES
            if not (self.raw_data_dir / name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing raw files: "
                + ", ".join(missing_files)
            )

    @staticmethod
    def _check_columns(
        dataframe: pd.DataFrame,
        required_columns: list[str],
        dataset_name: str,
    ) -> None:
        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing columns: "
                + ", ".join(missing_columns)
            )

    def validate_train(self, train: pd.DataFrame) -> None:
        required_columns = [
            "id",
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
        ]

        self._check_columns(
            train,
            required_columns,
            "train.csv",
        )

        if train.empty:
            raise ValueError("train.csv is empty.")

        if train["date"].isna().any():
            raise ValueError("train.csv contains missing dates.")

        if train["sales"].isna().any():
            raise ValueError("train.csv contains missing sales.")

        if (train["sales"] < 0).any():
            raise ValueError(
                "train.csv contains negative sales."
            )

    def validate_test(self, test: pd.DataFrame) -> None:
        self._check_columns(
            test,
            [
                "id",
                "date",
                "store_nbr",
                "family",
                "onpromotion",
            ],
            "test.csv",
        )

        if test.empty:
            raise ValueError("test.csv is empty.")

        if test["date"].isna().any():
            raise ValueError("test.csv contains missing dates.")

    def validate_stores(self, stores: pd.DataFrame) -> None:
        self._check_columns(
            stores,
            [
                "store_nbr",
                "city",
                "state",
                "type",
                "cluster",
            ],
            "stores.csv",
        )

        if stores["store_nbr"].duplicated().any():
            raise ValueError(
                "stores.csv contains duplicate store_nbr values."
            )

    def validate_oil(self, oil: pd.DataFrame) -> None:
        self._check_columns(
            oil,
            ["date", "dcoilwtico"],
            "oil.csv",
        )

        if oil["date"].isna().any():
            raise ValueError("oil.csv contains missing dates.")

    def validate_holidays(
        self,
        holidays: pd.DataFrame,
    ) -> None:
        self._check_columns(
            holidays,
            [
                "date",
                "type",
                "locale",
                "locale_name",
                "description",
                "transferred",
            ],
            "holidays_events.csv",
        )

        if holidays["date"].isna().any():
            raise ValueError(
                "holidays_events.csv contains missing dates."
            )

    def validate_transactions(
        self,
        transactions: pd.DataFrame,
    ) -> None:
        self._check_columns(
            transactions,
            ["date", "store_nbr", "transactions"],
            "transactions.csv",
        )

        if transactions["date"].isna().any():
            raise ValueError(
                "transactions.csv contains missing dates."
            )

        if (transactions["transactions"] < 0).any():
            raise ValueError(
                "transactions.csv contains negative transactions."
            )

    def validate_all(
        self,
        data: dict[str, pd.DataFrame],
    ) -> None:
        """Run every validation check."""
        self.validate_files_exist()

        self.validate_train(data["train"])
        self.validate_test(data["test"])
        self.validate_stores(data["stores"])
        self.validate_oil(data["oil"])
        self.validate_holidays(data["holidays"])
        self.validate_transactions(data["transactions"])
