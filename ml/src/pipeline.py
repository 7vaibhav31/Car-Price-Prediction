"""
pipeline.py — Build the full sklearn Pipeline:
              ColumnTransformer (encode + scale) → XGBRegressor.
"""
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import (
    NUMERIC_COLS, BRAND_COLS, OHE_COLS, PASSTHROUGH_COLS,
    PIPELINE_SAVE_PATH, RANDOM_STATE,
)


def build_pipeline() -> Pipeline:
    """
    Returns a fresh (unfitted) sklearn Pipeline with:
      - StandardScaler   → numeric columns
      - OrdinalEncoder   → brand  (handles unknown brands at inference)
      - OneHotEncoder    → fuel_type, transmission
      - XGBRegressor     → model
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_COLS,
            ),
            (
                "brand",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,         # unseen brand → -1
                ),
                BRAND_COLS,
            ),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",  # unseen category → all zeros
                    sparse_output=False,
                ),
                OHE_COLS,
            ),
        ],
        remainder="passthrough",   # keeps 'accident' as-is
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )
    return pipeline


def save_pipeline(pipeline: Pipeline, path: str = PIPELINE_SAVE_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved → {path}")


def load_pipeline(path: str = PIPELINE_SAVE_PATH) -> Pipeline:
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded ← {path}")
    return pipeline
