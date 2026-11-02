import glob
import os

import pandas as pd

from . import config


def download_dataset() -> str:
    import kagglehub

    return kagglehub.dataset_download(config.KAGGLE_DATASET)


def load_raw(path: str | None = None) -> pd.DataFrame:
    path = path or download_dataset()
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return pd.read_csv(csv_files[0])


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()
    return df
