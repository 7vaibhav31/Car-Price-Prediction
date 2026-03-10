"""
config.py — Central configuration for paths, constants, and column settings.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR          = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR     = os.path.join(BASE_DIR, "artifacts")

RAW_DATA_PATH     = os.path.join(DATA_DIR, "used_cars.csv")
PIPELINE_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "pipeline.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Column groups (used in ColumnTransformer) ────────────────────────
NUMERIC_COLS      = ["milage", "horse_power", "engine_litres", "car_age", "tx_speed"]
BRAND_COLS        = ["brand"]
OHE_COLS          = ["fuel_type", "transmission"]
PASSTHROUGH_COLS  = ["accident"]   # already binary int

TARGET_COL        = "price"

# ── Cleaning constants ────────────────────────────────────────────────
COLS_TO_DROP_INITIAL = ["model", "int_col", "clean_title"]
JUNK_FUEL_VALUES     = ["–", "not supported"]
CURRENT_YEAR         = 2026

# ── Train / Test split ────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
