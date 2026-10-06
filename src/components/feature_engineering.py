from __future__ import annotations

from pathlib import Path

import pandas as pd


class FeatureEngineering:
    """
    Create calendar, lag, rolling, store, transaction,
    oil-price and holiday features.
    """

    def __init__(self, processed_data_dir: str | Path):
        self.processed_data_dir = Path(processed_data_dir)

    @staticmethod
    def add_calendar_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        df = dataframe.copy()

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = (
            df["date"].dt.isocalendar().week.astype(int)
        )
        df["quarter"] = df["date"].dt.quarter

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["is_month_start"] = (
            df["date"].dt.is_month_start
        ).astype(int)

        df["is_month_end"] = (
            df["date"].dt.is_month_end
        ).astype(int)

        return df

    @staticmethod
    def add_lag_features(
        dataframe: pd.DataFrame,
        lags: tuple[int, ...] = (1, 7, 14, 28),
    ) -> pd.DataFrame:
        df = dataframe.copy()

        group_columns = ["store_nbr", "family"]

        for lag in lags:
            df[f"sales_lag_{lag}"] = (
                df.groupby(group_columns)["sales"]
                .shift(lag)
            )

        return df

    @staticmethod
    def add_rolling_features(
        dataframe: pd.DataFrame,
        windows: tuple[int, ...] = (7, 14, 28),
    ) -> pd.DataFrame:
        df = dataframe.copy()
        group_columns = ["store_nbr", "family"]

        # Shift first so today's target never enters today's features.
        shifted_sales = (
            df.groupby(group_columns)["sales"]
            .shift(1)
        )

        for window in windows:
            rolling_group = shifted_sales.groupby(
                [
                    df["store_nbr"],
                    df["family"],
                ]
            )

            df[f"sales_rolling_mean_{window}"] = (
                rolling_group.transform(
                    lambda values: values.rolling(window).mean()
                )
            )

            df[f"sales_rolling_std_{window}"] = (
                rolling_group.transform(
                    lambda values: values.rolling(window).std()
                )
            )

        return df

    @staticmethod
    def add_store_features(
        dataframe: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> pd.DataFrame:
        store_columns = [
            "store_nbr",
            "city",
            "state",
            "type",
            "cluster",
        ]

        store_info = (
            stores[store_columns]
            .drop_duplicates("store_nbr")
        )

        return dataframe.merge(
            store_info,
            on="store_nbr",
            how="left",
            validate="many_to_one",
        )

    @staticmethod
    def add_transaction_features(
        dataframe: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        transaction_info = (
            transactions[
                [
                    "date",
                    "store_nbr",
                    "transactions",
                ]
            ]
            .drop_duplicates(
                ["date", "store_nbr"]
            )
        )

        return dataframe.merge(
            transaction_info,
            on=["date", "store_nbr"],
            how="left",
            validate="many_to_one",
        )

    @staticmethod
    def add_oil_features(
        dataframe: pd.DataFrame,
        oil: pd.DataFrame,
    ) -> pd.DataFrame:
        df = dataframe.sort_values("date").copy()

        oil_info = (
            oil[["date", "dcoilwtico"]]
            .sort_values("date")
            .copy()
        )

        return pd.merge_asof(
            df,
            oil_info,
            on="date",
            direction="backward",
        )

    @staticmethod
    def add_holiday_features(
        dataframe: pd.DataFrame,
        holidays: pd.DataFrame,
    ) -> pd.DataFrame:
        holiday_info = holidays.copy()

        summary = (
            holiday_info.groupby("date")
            .agg(
                holiday_count=("type", "size"),
                holiday_types=(
                    "type",
                    lambda values: "|".join(
                        sorted(
                            values.astype(str).unique()
                        )
                    ),
                ),
            )
            .reset_index()
        )

        summary["is_holiday"] = (
            summary["holiday_count"] > 0
        ).astype(int)

        return dataframe.merge(
            summary,
            on="date",
            how="left",
            validate="many_to_one",
        )

    @staticmethod
    def fill_missing_feature_values(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        df = dataframe.copy()

        if "transactions" in df.columns:
            df["transactions"] = (
                df["transactions"].fillna(0)
            )

        if "dcoilwtico" in df.columns:
            df["dcoilwtico"] = (
                df["dcoilwtico"]
                .ffill()
                .bfill()
            )

        if "is_holiday" in df.columns:
            df["is_holiday"] = (
                df["is_holiday"].fillna(0)
            )

        if "holiday_count" in df.columns:
            df["holiday_count"] = (
                df["holiday_count"].fillna(0)
            )

        if "holiday_types" in df.columns:
            df["holiday_types"] = (
                df["holiday_types"]
                .fillna("None")
            )

        return df

    def build_train_features(
        self,
        train: pd.DataFrame,
        stores: pd.DataFrame,
        oil: pd.DataFrame,
        holidays: pd.DataFrame,
        transactions: pd.DataFrame,
        lags: tuple[int, ...] = (1, 7, 14, 28),
        rolling_windows: tuple[int, ...] = (7, 14, 28),
        save: bool = True,
    ) -> pd.DataFrame:
        """Build the complete leakage-safe training dataset."""

        df = train.copy()
        df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values(
            [
                "store_nbr",
                "family",
                "date",
            ]
        ).reset_index(drop=True)

        df = self.add_calendar_features(df)
        df = self.add_lag_features(df, lags)
        df = self.add_rolling_features(
            df,
            rolling_windows,
        )
        df = self.add_store_features(
            df,
            stores,
        )
        df = self.add_transaction_features(
            df,
            transactions,
        )
        df = self.add_oil_features(
            df,
            oil,
        )
        df = self.add_holiday_features(
            df,
            holidays,
        )
        df = self.fill_missing_feature_values(df)

        required_history_columns = [
            f"sales_lag_{lag}"
            for lag in lags
        ]

        for window in rolling_windows:
            required_history_columns.extend(
                [
                    f"sales_rolling_mean_{window}",
                    f"sales_rolling_std_{window}",
                ]
            )

        # These rows do not yet have enough historical information
        # to calculate the requested lag/rolling features.
        df = df.dropna(
            subset=required_history_columns
        ).reset_index(drop=True)

        if save:
            self.processed_data_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                self.processed_data_dir
                / "train_featured.csv"
            )

            df.to_csv(
                output_path,
                index=False,
            )

        return df
