from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.exception import ForecastingException
from src.logger import logger


class ForecastPipeline:
    """
    Load a trained model and generate predictions from a prepared
    feature dataset.

    Note:
    The input must already contain the same feature columns used
    during training. Building true future recursive lag features
    is handled separately from this prediction wrapper.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = Path(project_root)

        self.model_path = (
            self.project_root
            / "models"
            / "best_forecasting_model.joblib"
        )

        self.artifacts_dir = (
            self.project_root
            / "artifacts"
        )

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        logger.info(
            "Loading model from %s",
            self.model_path,
        )

        return joblib.load(
            self.model_path
        )

    @staticmethod
    def prepare_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = [
            "date",
            "store_nbr",
            "family",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing forecast columns: "
                + ", ".join(
                    missing_columns
                )
            )

        return dataframe.drop(
            columns=[
                "sales",
                "date",
                "holiday_types",
            ],
            errors="ignore",
        )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            if dataframe.empty:
                raise ValueError(
                    "Forecast dataframe is empty."
                )

            model = self.load_model()

            X = self.prepare_features(
                dataframe
            )

            predictions = model.predict(X)

            result = dataframe[
                [
                    "date",
                    "store_nbr",
                    "family",
                ]
            ].copy()

            result[
                "predicted_sales"
            ] = predictions

            result[
                "predicted_sales"
            ] = result[
                "predicted_sales"
            ].clip(lower=0)

            return result

        except Exception as exc:
            logger.exception(
                "Forecast prediction failed."
            )
            raise ForecastingException(
                str(exc)
            ) from exc

    def predict_from_csv(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Forecast input not found: {input_path}"
            )

        dataframe = pd.read_csv(
            input_path,
            parse_dates=["date"],
        )

        predictions = self.predict(
            dataframe
        )

        if output_path is None:
            output_path = (
                self.artifacts_dir
                / "forecast_predictions.csv"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Predictions saved to %s",
            output_path,
        )

        return predictions
