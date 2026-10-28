import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def find_dataset_path(file_path: str = None) -> str:
    """Find auto-mpg.csv across standard project directories."""
    if file_path and os.path.exists(file_path):
        return file_path

    # Candidate locations relative to current script and working dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        file_path,
        os.path.join(base_dir, "..", "data", "auto-mpg.csv"),
        os.path.join(base_dir, "..", "auto-mpg.csv"),
        os.path.join(base_dir, "data", "auto-mpg.csv"),
        "project/data/auto-mpg.csv",
        "project/auto-mpg.csv",
        "data/auto-mpg.csv",
        "auto-mpg.csv",
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError("Could not find auto-mpg.csv in standard project paths.")


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing '?' in horsepower using cylinder-grouped median with global fallback, placing 'car name' first."""
    cleaned = df.copy()
    cleaned["horsepower"] = pd.to_numeric(cleaned["horsepower"].replace("?", np.nan))
    cylinder_median = cleaned.groupby("cylinders")["horsepower"].transform("median")
    cleaned["horsepower"] = cleaned["horsepower"].fillna(cylinder_median).fillna(cleaned["horsepower"].median())
    if "car name" in cleaned.columns:
        cols = ["car name"] + [c for c in cleaned.columns if c != "car name"]
        cleaned = cleaned[cols]
    return cleaned


def load_and_clean_data(file_path: str = None) -> pd.DataFrame:
    """Load Auto MPG dataset and clean missing values."""
    resolved_path = find_dataset_path(file_path)
    df = pd.read_csv(resolved_path)
    return clean_missing_values(df)


def get_feature_target_split(df: pd.DataFrame):
    """Separate predictor features from target mpg."""
    features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
        "origin",
    ]
    X = df[features].copy()
    y = df["mpg"].copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets with fixed random seed for reproducibility."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
