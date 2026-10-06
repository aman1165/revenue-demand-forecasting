def test_training_pipeline_import():
    from src.pipeline.train_pipeline import TrainingPipeline

    assert TrainingPipeline is not None


def test_forecast_pipeline_import():
    from src.pipeline.forecast_pipeline import ForecastPipeline

    assert ForecastPipeline is not None


def test_component_imports():
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.feature_engineering import FeatureEngineering
    from src.components.forecasting import Forecasting
    from src.components.model_trainer import ModelTrainer

    assert DataIngestion is not None
    assert DataValidation is not None
    assert FeatureEngineering is not None
    assert Forecasting is not None
    assert ModelTrainer is not None