from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_forecasting_model.joblib"
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "train_featured.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(FEATURE_FILE, nrows=1)

    X = df.drop(
        columns=["sales", "date", "holiday_types"],
        errors="ignore"
    )

    prediction = model.predict(X)

    assert len(prediction) == 1
    assert prediction[0] >= 0