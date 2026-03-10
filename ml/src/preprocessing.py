import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH, COLS_TO_DROP_INITIAL, JUNK_FUEL_VALUES,
    CURRENT_YEAR, TARGET_COL, TEST_SIZE, RANDOM_STATE,
)
"""
preprocessing.py — Raw data cleaning ONLY.
Returns a clean X, y ready for the sklearn Pipeline.
No encoding / scaling here — those live inside the pipeline.
"""


# ─────────────────────────────────────────────────────────────────────
def _categorize_transmission(t: str) -> str:
    t = str(t).lower()
    if "auto" in t or "a/t" in t:  return "automatic"
    if "manual" in t or "m/t" in t: return "manual"
    if "cvt" in t or "variable" in t: return "cvt"
    if "dual" in t or "dct" in t:  return "dual_clutch"
    return "other"


def _extract_tx_speed(t: str) -> int:
    match = re.search(r"(\d+)-Speed", str(t), re.IGNORECASE)
    return int(match.group(1)) if match else 6  # fallback = dataset median


# ─────────────────────────────────────────────────────────────────────
def clean(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  shape: {df.shape}")

    # ── Drop unneeded columns
    df.drop(columns=COLS_TO_DROP_INITIAL, errors="ignore", inplace=True)

    # ── Mileage: "51,000 mi." → 51000.0
    df["milage"] = (
        df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(float)
    )

    # ── Price: "$10,300" → 10300.0
    df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)

    # ── Accident → binary
    df["accident"] = (
        df["accident"]
        .apply(lambda x: 1 if x == "At least 1 accident or damage reported" else 0)
    )

    # ── Engine features
    df["horse_power"]   = pd.to_numeric(
        df["engine"].str.extract(r"(\d+\.?\d*)HP")[0], errors="coerce"
    )
    df["engine_litres"] = pd.to_numeric(
        df["engine"].str.extract(r"(\d+\.?\d*)L")[0], errors="coerce"
    )
    df.drop(columns=["engine"], inplace=True)

    # ── Transmission: speed + category
    df["tx_speed"]      = df["transmission"].apply(_extract_tx_speed)
    df["transmission"]  = df["transmission"].apply(_categorize_transmission)

    # ── Fuel type: clean junk → impute by brand mode
    df["fuel_type"] = df["fuel_type"].astype(object).replace(JUNK_FUEL_VALUES, np.nan)
    df["fuel_type"] = df.groupby("brand")["fuel_type"].transform(
        lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan)
    )
    df["fuel_type"] = df["fuel_type"].fillna(df["fuel_type"].mode()[0])

    # ── Numeric imputation (by brand median, then global median)
    for col in ["horse_power", "engine_litres"]:
        df[col] = df.groupby("brand")[col].transform(
            lambda x: x.fillna(x.median() if not x.isna().all() else np.nan)
        )
        df[col] = df[col].fillna(df[col].median())

    # ── Car age feature
    df["car_age"] = CURRENT_YEAR - df["model_year"]

    # ── Drop remaining raw columns
    df.drop(columns=["model_year", "ext_col"], errors="ignore", inplace=True)

    print(f"[clean] shape: {df.shape} | nulls: {df.isnull().sum().sum()}")
    return df


def get_splits():
    """Load, clean, split → X_train, X_test, y_train, y_test."""
    df = clean()
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_splits()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(X_train.dtypes)
