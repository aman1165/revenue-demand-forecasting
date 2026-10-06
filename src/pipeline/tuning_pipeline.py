from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.components.model_tuning import ModelTuner
from src.exception import ForecastingException
from src.logger import logger
from src.utils import load_config


class TuningPipeline:
    """Run model hyperparameter tuning after the initial training pipeline."""

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

        paths = config.get("paths", {})
        self.processed_data_dir = (
            self.project_root
            / paths.get("processed_data", "data/processed")
        )
        self.models_dir = (
            self.project_root
            / paths.get("models", "models")
        )
        self.artifacts_dir = (
            self.project_root
            / paths.get("artifacts", "artifacts")
        )

        self.random_state = int(
            config.get("project", {})
            .get("random_state", 42)
        )

    def run(self) -> dict:
        try:
            logger.info("========== TUNING PIPELINE STARTED ==========")

            featured_data_path = (
                self.processed_data_dir
                / "train_featured.csv"
            )

            if not featured_data_path.exists():
                raise FileNotFoundError(
                    f"Featured dataset not found: {featured_data_path}. "
                    "Run the training pipeline first."
                )

            featured_data = pd.read_csv(
                featured_data_path,
                parse_dates=["date"],
            )

            logger.info(
                "Featured data loaded successfully: %s",
                featured_data.shape,
            )

            tuner = ModelTuner(
                models_dir=self.models_dir,
                artifacts_dir=self.artifacts_dir,
                random_state=self.random_state,
            )

            result = tuner.tune(
                dataframe=featured_data,
                target_column=config_target(self.config_path),
            )

            logger.info(
                "Tuning result: improved=%s, selected_source=%s",
                result["improved"],
                result["selected_source"],
            )
            logger.info("========== TUNING PIPELINE FINISHED ==========")

            return result

        except Exception as exc:
            logger.exception("Tuning pipeline failed.")
            raise ForecastingException(str(exc)) from exc


def config_target(config_path: str | Path) -> str:
    """Read the target column from the project config."""
    config = load_config(config_path)
    return (
        config.get("data", {})
        .get("target", "sales")
    )


if __name__ == "__main__":
    pipeline = TuningPipeline()
    result = pipeline.run()

    print("\nTuning completed.")
    print(
        f"Best candidate: {result['best_candidate_name']}"
    )
    print(
        f"Tuned validation metrics: "
        f"{result['tuned_validation_metrics']}"
    )
    print(
        f"Improved existing model: {result['improved']}"
    )
    print(
        f"Final test metrics: "
        f"{result['final_test_metrics']}"
    )
    print(
        f"Production model: {result['model_path']}"
    )
