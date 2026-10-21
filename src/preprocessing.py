"""Shared data validation and preprocessing for model training and inference."""
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(file_path=DATA_PATH):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {path}. Provide the original diabetes CSV.")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"Dataset contains no rows: {path}")
    return data


def clean_data(data):
    data = data.copy()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")
    missing = [name for name in FEATURE_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"Required feature columns are missing: {missing}")
    data = data.drop_duplicates()
    for column in ZERO_IS_MISSING:
        data[column] = data[column].replace(0, float("nan"))
    return data


def prepare_data(data):
    data = clean_data(data)
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only binary values 0 and 1, with no missing values.")
    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if X.isna().all(axis=0).any():
        raise ValueError("At least one required feature contains no usable numeric values.")
    return X, y.astype("int64")


def create_preprocessing_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def split_data(X, y):
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("At least two rows from each outcome class are required for a stratified split.")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
