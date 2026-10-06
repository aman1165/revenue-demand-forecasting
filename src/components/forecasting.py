from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class ForecastMetrics:
    """Standard regression metrics used for forecasting evaluation."""

    mae: float
    rmse: float
    mape: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "R2": self.r2,
        }


class Forecasting:
    """Baseline forecasts and evaluation helpers."""

    @staticmethod
    def safe_mape(
        actual,
        predicted,
    ) -> float:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        non_zero = actual != 0

        if not non_zero.any():
            return 0.0

        return float(
            np.mean(
                np.abs(
                    (
                        actual[non_zero]
                        - predicted[non_zero]
                    )
                    / actual[non_zero]
                )
            )
            * 100
        )

    @classmethod
    def evaluate(
        cls,
        actual,
        predicted,
    ) -> ForecastMetrics:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        return ForecastMetrics(
            mae=float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            rmse=float(rmse),
            mape=cls.safe_mape(
                actual,
                predicted,
            ),
            r2=float(
                r2_score(
                    actual,
                    predicted,
                )
            ),
        )

    @staticmethod
    def previous_day_baseline(
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        return dataframe["sales_lag_1"]

    @staticmethod
    def previous_week_baseline(
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        return dataframe["sales_lag_7"]

    @classmethod
    def evaluate_baseline(
        cls,
        dataframe: pd.DataFrame,
        prediction_column: str,
    ) -> ForecastMetrics:
        prediction = dataframe[prediction_column]
        actual = dataframe["sales"]

        valid_rows = (
            actual.notna()
            & prediction.notna()
        )

        return cls.evaluate(
            actual.loc[valid_rows],
            prediction.loc[valid_rows],
        )

    @staticmethod
    def compare_models(
        model_results: list[dict],
    ) -> pd.DataFrame:
        if not model_results:
            raise ValueError(
                "No model results were supplied."
            )

        return (
            pd.DataFrame(model_results)
            .sort_values("MAE")
            .reset_index(drop=True)
        )

    @staticmethod
    def daily_summary(
        dates,
        actual,
        predicted,
    ) -> pd.DataFrame:
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "actual": actual,
                "predicted": predicted,
            }
        )

        return (
            daily.groupby(
                "date",
                as_index=False,
            )
            .agg(
                actual=("actual", "sum"),
                predicted=("predicted", "sum"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
