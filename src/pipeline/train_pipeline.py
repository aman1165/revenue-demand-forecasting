from __future__ import annotations

from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.exception import ForecastingException
from src.logger import logger
from src.utils import load_config


class TrainingPipeline:
    """Run the complete training workflow from raw data to model."""

    def __init__(
        self,
        project_root: str | Path | None = None,
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = Path(project_root)
        self.config_path = (
            self.project_root
            / "config"
            / "config.yaml"
        )

        config = load_config(self.config_path)

        self.raw_data_dir = (
            self.project_root
            / config.get(
                "paths",
                {},
            ).get(
                "raw_data",
                "data/raw",
            )
        )

        self.processed_data_dir = (
            self.project_root
            / config.get(
                "paths",
                {},
            ).get(
                "processed_data",
                "data/processed",
            )
        )

        self.models_dir = (
            self.project_root
            / config.get(
                "paths",
                {},
            ).get(
                "models",
                "models",
            )
        )

    def run(self) -> dict:
        try:
            logger.info(
                "========== TRAINING STARTED =========="
            )

            ingestion = DataIngestion(
                self.raw_data_dir
            )

            data = ingestion.load_all()

            logger.info(
                "Raw data loaded successfully."
            )

            validation = DataValidation(
                self.raw_data_dir
            )
            validation.validate_all(data)

            logger.info(
                "Data validation completed."
            )

            config = load_config(
                self.config_path
            )

            feature_config = config.get(
                "features",
                {},
            )

            lags = tuple(
                feature_config.get(
                    "lags",
                    [1, 7, 14, 28],
                )
            )

            rolling_windows = tuple(
                feature_config.get(
                    "rolling_windows",
                    [7, 14, 28],
                )
            )

            feature_engineering = (
                FeatureEngineering(
                    self.processed_data_dir
                )
            )

            featured_data = (
                feature_engineering.build_train_features(
                    train=data["train"],
                    stores=data["stores"],
                    oil=data["oil"],
                    holidays=data["holidays"],
                    transactions=data["transactions"],
                    lags=lags,
                    rolling_windows=rolling_windows,
                )
            )

            logger.info(
                "Feature dataset created: %s",
                featured_data.shape,
            )

            model_trainer = ModelTrainer(
                self.models_dir,
                random_state=config.get(
                    "project",
                    {},
                ).get(
                    "random_state",
                    42,
                ),
            )

            result = (
                model_trainer.train_and_compare(
                    featured_data,
                    target_column=config.get(
                        "data",
                        {},
                    ).get(
                        "target",
                        "sales",
                    ),
                )
            )

            logger.info(
                "Best model: %s",
                result["best_model_name"],
            )
            logger.info(
                "Test metrics: %s",
                result["test_metrics"],
            )
            logger.info(
                "========== TRAINING FINISHED =========="
            )

            return result

        except Exception as exc:
            logger.exception(
                "Training pipeline failed."
            )
            raise ForecastingException(
                str(exc)
            ) from exc


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run()

    print("\nTraining completed.")
    print(
        f"Best model: "
        f"{result['best_model_name']}"
    )
    print(
        f"Test metrics: "
        f"{result['test_metrics']}"
    )
    print(
        f"Saved model: "
        f"{result['model_path']}"
    )
