from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_raw_files_exist():
    required_files = [
        "train.csv",
        "test.csv",
        "stores.csv",
        "oil.csv",
        "holidays_events.csv",
        "transactions.csv",
    ]

    for file_name in required_files:
        assert (RAW_DIR / file_name).exists()


def test_feature_dataset_exists():
    feature_file = PROCESSED_DIR / "train_featured.csv"

    assert feature_file.exists()


def test_feature_dataset_has_required_columns():
    feature_file = PROCESSED_DIR / "train_featured.csv"

    df = pd.read_csv(feature_file, nrows=5)

    required_columns = [
        "id",
        "store_nbr",
        "family",
        "sales",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
    ]

    for column in required_columns:
        assert column in df.columns