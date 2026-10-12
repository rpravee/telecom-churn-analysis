"""Loading, cleaning and (stateless) feature engineering for the Telco churn data."""
from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ENGINEERED_COLS = ["n_addons", "is_autopay"]


def load_raw(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and the hidden blank-string nulls in TotalCharges.

    The 11 blank TotalCharges rows all have tenure == 0: brand-new customers who have not
    been billed yet. Their true total is 0, so we fill with 0 instead of dropping them.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    not_yet_billed = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[not_yet_billed, "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().any():
        raise ValueError("Unexpected missing TotalCharges for customers with tenure > 0")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):  # works for object and pandas>=3 string dtype
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (no statistics learned from data), so no leakage risk."""
    df = df.copy()
    df["n_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df


def load_and_prepare(path: Path | str = DEFAULT_PATH, engineered: bool = True) -> pd.DataFrame:
    df = clean(load_raw(path))
    return add_features(df) if engineered else df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]
    return X, y
