from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.components.model_trainer import ModelTrainer
from src.logger import logger


class ModelTuner:
    """
    Tune the best candidate selected by ModelTrainer.

    The tuning process is deliberately separate from initial model training.
    Hyperparameters are searched only on the chronological training period,
    then the tuned model is compared on the untouched validation period.

    If the tuned model improves validation MAE, it is refit on Train +
    Validation and overwrites best_forecasting_model.joblib.
    """

    def __init__(
        self,
        models_dir: str | Path,
        artifacts_dir: str | Path,
        random_state: int = 42,
        max_tuning_rows: int = 300_000,
        n_iter: int = 12,
        n_splits: int = 3,
    ):
        self.models_dir = Path(models_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.random_state = random_state
        self.max_tuning_rows = max_tuning_rows
        self.n_iter = n_iter
        self.n_splits = n_splits

        self.model_path = (
            self.models_dir
            / "best_forecasting_model.joblib"
        )

    @staticmethod
    def _date_based_splits(
        dataframe: pd.DataFrame,
        n_splits: int = 3,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Create chronological folds without splitting the same date across
        train and validation rows.
        """
        if "date" not in dataframe.columns:
            raise ValueError("'date' column is required for tuning.")

        df = dataframe.sort_values("date").reset_index(drop=True)
        unique_dates = pd.Index(
            pd.to_datetime(df["date"]).dropna().sort_values().unique()
        )

        if len(unique_dates) < n_splits + 1:
            raise ValueError(
                f"Need at least {n_splits + 1} unique dates for tuning CV. "
                f"Found {len(unique_dates)}."
            )

        date_blocks = np.array_split(unique_dates, n_splits + 1)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        # Expanding-window folds:
        # fold 1 -> block 0 train, block 1 validation
        # fold 2 -> blocks 0-1 train, block 2 validation
        # fold 3 -> blocks 0-2 train, block 3 validation
        for fold in range(1, n_splits + 1):
            train_dates = np.concatenate(date_blocks[:fold])
            validation_dates = date_blocks[fold]

            train_mask = df["date"].isin(train_dates).to_numpy()
            validation_mask = df["date"].isin(validation_dates).to_numpy()

            train_indices = np.flatnonzero(train_mask)
            validation_indices = np.flatnonzero(validation_mask)

            if len(train_indices) == 0 or len(validation_indices) == 0:
                continue

            splits.append((train_indices, validation_indices))

        if len(splits) < 2:
            raise ValueError("Unable to create enough chronological CV folds.")

        return splits

    def _build_model_and_search_space(
        self,
        model_name: str,
    ) -> tuple[Any, dict[str, Iterable[Any]]]:
        """Return the estimator and search space for the selected model."""
        if model_name == "XGBoost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "XGBoost is selected as the best candidate, but xgboost "
                    "is not installed. Run: pip install xgboost"
                ) from exc

            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=self.random_state,
            )

            search_space = {
                "model__n_estimators": [300, 500, 700],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
                "model__max_depth": [5, 7, 9],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            }
            return model, search_space

        if model_name == "HistGradientBoosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(
                random_state=self.random_state
            )

            search_space = {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
                "model__max_iter": [200, 300, 400, 500],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [20, 50, 100],
                "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
            }
            return model, search_space

        if model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(
                n_jobs=-1,
                random_state=self.random_state,
            )

            search_space = {
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [12, 16, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7, 1.0],
            }
            return model, search_space

        raise ValueError(
            f"Unsupported model for tuning: {model_name}"
        )

    def _prepare_tuning_data(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep the latest chronological rows inside training only."""
        train_df = (
            train_df.sort_values("date")
            .reset_index(drop=True)
        )

        if len(train_df) <= self.max_tuning_rows:
            return train_df

        return train_df.iloc[-self.max_tuning_rows :].copy()

    def tune(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "sales",
    ) -> dict[str, Any]:
        """Tune, compare, optionally refit and save the final best model."""
        if dataframe.empty:
            raise ValueError("Tuning dataframe is empty.")

        logger.info("========== MODEL TUNING STARTED ==========")

        trainer = ModelTrainer(
            models_dir=self.models_dir,
            random_state=self.random_state,
        )

        train_df, validation_df, test_df = (
            trainer.chronological_split(dataframe)
        )

        # The current saved model was selected by ModelTrainer before tuning.
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Existing best model not found: {self.model_path}. "
                "Run the training pipeline first."
            )

        existing_model = joblib.load(self.model_path)

        X_train, y_train = trainer.prepare_xy(
            train_df,
            target_column,
        )
        X_validation, y_validation = trainer.prepare_xy(
            validation_df,
            target_column,
        )
        X_test, y_test = trainer.prepare_xy(
            test_df,
            target_column,
        )

        existing_validation_prediction = existing_model.predict(
            X_validation
        )
        existing_validation_metrics = trainer.evaluate(
            y_validation,
            existing_validation_prediction,
        )

        validation_comparison_path = (
            self.models_dir
            / "validation_model_comparison.csv"
        )

        if validation_comparison_path.exists():
            comparison = pd.read_csv(validation_comparison_path)
            if "Model" in comparison.columns:
                best_candidate_name = comparison.iloc[
                    comparison["MAE"].astype(float).argmin()
                ]["Model"]
            else:
                best_candidate_name = None
        else:
            best_candidate_name = None

        if best_candidate_name not in {
            "XGBoost",
            "HistGradientBoosting",
            "RandomForest",
        }:
            # The actual loaded best model may come from any one of the
            # three supported trainer models. Prefer the model that won the
            # previous validation comparison when that file is available.
            best_candidate_name = type(
                existing_model.named_steps.get("model")
            ).__name__.replace("Regressor", "")

            if best_candidate_name == "RandomForest":
                pass
            elif best_candidate_name == "HistGradientBoosting":
                pass
            elif best_candidate_name == "XGB":
                best_candidate_name = "XGBoost"
            else:
                raise ValueError(
                    "Could not determine which model should be tuned."
                )

        logger.info(
            "Best candidate selected for tuning: %s",
            best_candidate_name,
        )

        tuning_df = self._prepare_tuning_data(train_df)
        X_tune, y_tune = trainer.prepare_xy(
            tuning_df,
            target_column,
        )

        # Build chronological indices from the tuning frame.
        cv_splits = self._date_based_splits(
            tuning_df,
            n_splits=self.n_splits,
        )

        model, search_space = self._build_model_and_search_space(
            best_candidate_name
        )

        preprocessor = trainer.build_preprocessor(X_tune)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=self.n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv_splits,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2,
            return_train_score=False,
        )

        logger.info(
            "Tuning started on %s rows with %s CV folds and %s iterations.",
            len(tuning_df),
            len(cv_splits),
            self.n_iter,
        )

        search.fit(X_tune, y_tune)

        tuned_model = search.best_estimator_
        tuned_validation_prediction = tuned_model.predict(
            X_validation
        )
        tuned_validation_metrics = trainer.evaluate(
            y_validation,
            tuned_validation_prediction,
        )

        old_mae = float(existing_validation_metrics["MAE"])
        tuned_mae = float(tuned_validation_metrics["MAE"])

        improved = tuned_mae < old_mae

        logger.info(
            "Existing validation MAE: %.6f",
            old_mae,
        )
        logger.info(
            "Tuned validation MAE: %.6f",
            tuned_mae,
        )

        final_model = existing_model
        selected_source = "existing"

        if improved:
            logger.info(
                "Tuned model improved validation MAE. Refitting on Train + Validation."
            )

            X_train_validation = pd.concat(
                [X_train, X_validation],
                axis=0,
                ignore_index=True,
            )
            y_train_validation = pd.concat(
                [y_train, y_validation],
                axis=0,
                ignore_index=True,
            )

            tuned_model.fit(
                X_train_validation,
                y_train_validation,
            )

            final_model = tuned_model
            selected_source = "tuned"

            # Intentionally overwrite the previous production model.
            joblib.dump(final_model, self.model_path)
            logger.info(
                "Overwrote production model: %s",
                self.model_path,
            )
        else:
            logger.info(
                "Tuned model did not improve validation MAE. Existing model retained."
            )

        final_test_prediction = final_model.predict(X_test)
        final_test_prediction = np.clip(
            final_test_prediction,
            0,
            None,
        )

        final_test_metrics = trainer.evaluate(
            y_test,
            final_test_prediction,
        )

        tuning_results = pd.DataFrame(
            search.cv_results_
        ).copy()
        tuning_results["mean_test_MAE"] = -tuning_results[
            "mean_test_score"
        ]
        tuning_results = tuning_results.sort_values(
            "mean_test_MAE"
        ).reset_index(drop=True)
        tuning_results.to_csv(
            self.artifacts_dir / "model_tuning_results.csv",
            index=False,
        )

        validation_comparison = pd.DataFrame(
            [
                {
                    "Model": f"Existing Best - {best_candidate_name}",
                    **existing_validation_metrics,
                },
                {
                    "Model": f"Tuned - {best_candidate_name}",
                    **tuned_validation_metrics,
                },
            ]
        ).sort_values("MAE").reset_index(drop=True)

        validation_comparison.to_csv(
            self.artifacts_dir / "tuned_validation_comparison.csv",
            index=False,
        )

        pd.DataFrame(
            [final_test_metrics]
        ).to_csv(
            self.artifacts_dir / "tuned_final_test_metrics.csv",
            index=False,
        )

        test_predictions = test_df[
            ["date", "store_nbr", "family"]
        ].copy()
        test_predictions["actual_sales"] = y_test.to_numpy()
        test_predictions["predicted_sales"] = final_test_prediction
        test_predictions.to_csv(
            self.artifacts_dir / "tuned_test_predictions.csv",
            index=False,
        )

        logger.info("Final test metrics: %s", final_test_metrics)
        logger.info("========== MODEL TUNING FINISHED ==========")

        return {
            "best_candidate_name": best_candidate_name,
            "best_params": search.best_params_,
            "best_cv_mae": float(-search.best_score_),
            "existing_validation_metrics": existing_validation_metrics,
            "tuned_validation_metrics": tuned_validation_metrics,
            "improved": improved,
            "selected_source": selected_source,
            "final_test_metrics": final_test_metrics,
            "model_path": self.model_path,
        }
