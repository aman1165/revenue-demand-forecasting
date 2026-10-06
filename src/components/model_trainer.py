from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class ModelTrainer:
    """
    Train tree-based forecasting models with a chronological split.

    Ordinal encoding is used instead of large one-hot matrices so the
    Store Sales dataset remains practical on a local machine.
    """

    def __init__(
        self,
        models_dir: str | Path,
        random_state: int = 42,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.random_state = random_state

    @staticmethod
    def chronological_split(
        dataframe: pd.DataFrame,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if train_ratio + validation_ratio >= 1:
            raise ValueError(
                "train_ratio + validation_ratio must be < 1."
            )

        df = dataframe.sort_values(
            "date"
        ).reset_index(drop=True)

        dates = sorted(
            df["date"].dropna().unique()
        )

        if len(dates) < 3:
            raise ValueError(
                "At least 3 unique dates are required."
            )

        train_end = max(
            1,
            int(len(dates) * train_ratio),
        )

        validation_end = max(
            train_end + 1,
            int(
                len(dates)
                * (train_ratio + validation_ratio)
            ),
        )

        validation_end = min(
            validation_end,
            len(dates) - 1,
        )

        train_dates = dates[:train_end]
        validation_dates = dates[
            train_end:validation_end
        ]
        test_dates = dates[
            validation_end:
        ]

        train_df = df[
            df["date"].isin(train_dates)
        ].copy()

        validation_df = df[
            df["date"].isin(validation_dates)
        ].copy()

        test_df = df[
            df["date"].isin(test_dates)
        ].copy()

        return (
            train_df,
            validation_df,
            test_df,
        )

    @staticmethod
    def prepare_xy(
        dataframe: pd.DataFrame,
        target_column: str = "sales",
    ) -> tuple[pd.DataFrame, pd.Series]:
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        features = dataframe.drop(
            columns=[
                target_column,
                "date",
                "holiday_types",
            ],
            errors="ignore",
        ).copy()

        target = dataframe[
            target_column
        ].copy()

        return features, target

    @staticmethod
    def build_preprocessor(
        X: pd.DataFrame,
    ) -> ColumnTransformer:
        categorical_columns = (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            )
            .columns
            .tolist()
        )

        numeric_columns = (
            X.select_dtypes(
                exclude=[
                    "object",
                    "category",
                ]
            )
            .columns
            .tolist()
        )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

    def build_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {
            "HistGradientBoosting": (
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                )
            ),
            "RandomForest": (
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=self.random_state,
                )
            ),
        }

        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=self.random_state,
            )
        except ImportError:
            pass

        return models

    @staticmethod
    def evaluate(
        actual: pd.Series,
        predicted,
    ) -> dict[str, float]:
        actual = np.asarray(
            actual,
            dtype=float,
        )
        predicted = np.asarray(
            predicted,
            dtype=float,
        )

        non_zero = actual != 0

        if non_zero.any():
            mape = (
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
        else:
            mape = 0.0

        return {
            "MAE": float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                )
            ),
            "MAPE": float(mape),
            "R2": float(
                r2_score(
                    actual,
                    predicted,
                )
            ),
        }

    def train_and_compare(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "sales",
    ) -> dict:
        train_df, validation_df, test_df = (
            self.chronological_split(dataframe)
        )

        X_train, y_train = self.prepare_xy(
            train_df,
            target_column,
        )
        X_validation, y_validation = self.prepare_xy(
            validation_df,
            target_column,
        )
        X_test, y_test = self.prepare_xy(
            test_df,
            target_column,
        )

        preprocessor = self.build_preprocessor(
            X_train
        )

        results = []
        trained_models = {}

        for model_name, model in self.build_models().items():
            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            validation_prediction = (
                pipeline.predict(X_validation)
            )

            metrics = self.evaluate(
                y_validation,
                validation_prediction,
            )

            results.append(
                {
                    "Model": model_name,
                    **metrics,
                }
            )

            trained_models[
                model_name
            ] = pipeline

        comparison = (
            pd.DataFrame(results)
            .sort_values("MAE")
            .reset_index(drop=True)
        )

        best_model_name = comparison.iloc[
            0
        ]["Model"]

        best_model = trained_models[
            best_model_name
        ]

        test_prediction = best_model.predict(
            X_test
        )

        test_metrics = self.evaluate(
            y_test,
            test_prediction,
        )

        model_path = (
            self.models_dir
            / "best_forecasting_model.joblib"
        )

        joblib.dump(
            best_model,
            model_path,
        )

        prediction_output = test_df[
            [
                "date",
                "store_nbr",
                "family",
            ]
        ].copy()

        prediction_output[
            "actual_sales"
        ] = y_test.to_numpy()

        prediction_output[
            "predicted_sales"
        ] = np.clip(
            test_prediction,
            0,
            None,
        )

        prediction_path = (
            self.models_dir
            / "test_predictions.csv"
        )

        prediction_output.to_csv(
            prediction_path,
            index=False,
        )

        comparison_path = (
            self.models_dir
            / "validation_model_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        metrics_path = (
            self.models_dir
            / "final_test_metrics.csv"
        )

        pd.DataFrame(
            [test_metrics]
        ).to_csv(
            metrics_path,
            index=False,
        )

        return {
            "comparison": comparison,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "test_metrics": test_metrics,
            "model_path": model_path,
            "prediction_path": prediction_path,
        }
