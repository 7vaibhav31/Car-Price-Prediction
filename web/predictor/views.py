"""
predictor/views.py — Load the pipeline and serve predictions.
"""
import pickle
from pathlib import Path

import pandas as pd
from django.shortcuts import render

# ── Resolve path to the saved pipeline from any working directory ──────────
# web/predictor/views.py  →  .parents[2] = car_prediction_model/
_REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = _REPO_ROOT / "ml" / "artifacts" / "pipeline.pkl"

_pipeline = None   # lazy-loaded once on first request


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        with open(PIPELINE_PATH, "rb") as f:
            _pipeline = pickle.load(f)
    return _pipeline


# ── Static option lists ────────────────────────────────────────────────────
BRANDS = sorted([
    "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz",
    "Audi", "Hyundai", "Kia", "Nissan", "Jeep", "Ram", "GMC",
    "Volkswagen", "Subaru", "Mazda", "Lexus", "Dodge", "Chrysler",
    "Tesla", "Volvo", "Land Rover", "Porsche", "Cadillac", "Buick",
    "Lincoln", "Infiniti", "Acura", "Mitsubishi", "Genesis",
])

FUEL_TYPES = [
    "Gasoline", "Diesel", "Hybrid", "Electric",
    "Plug-In Hybrid", "E85 Flex Fuel", "Hydrogen",
]

TRANSMISSIONS = [
    ("automatic",   "Automatic"),
    ("manual",      "Manual"),
    ("cvt",         "CVT"),
    ("dual_clutch", "Dual Clutch"),
    ("other",       "Other"),
]


def _make_options(choices, selected_val):
    """Return list of (value, label, is_selected) for use in templates."""
    result = []
    for item in choices:
        if isinstance(item, tuple):
            val, label = item
        else:
            val = label = item
        result.append({"value": val, "label": label, "selected": val == selected_val})
    return result


def predict_view(request):
    fd = {}           # form_data shorthand
    result = None
    error  = None

    if request.method == "POST":
        fd = request.POST.dict()
        try:
            input_dict = {
                "brand":         fd.get("brand", "Toyota"),
                "milage":        float(fd.get("milage", 0)),
                "fuel_type":     fd.get("fuel_type", "Gasoline"),
                "transmission":  fd.get("transmission", "automatic"),
                "accident":      int(fd.get("accident", 0)),
                "tx_speed":      int(fd.get("tx_speed", 6)),
                "horse_power":   float(fd.get("horse_power", 150)),
                "engine_litres": float(fd.get("engine_litres", 2.0)),
                "car_age":       int(fd.get("car_age", 5)),
            }
            pipeline = _load_pipeline()
            X = pd.DataFrame([input_dict])
            price = pipeline.predict(X)[0]
            result = f"${price:,.2f}"
        except FileNotFoundError:
            error = (
                "Pipeline not found. Please run <code>python train.py</code> "
                "inside <code>ml/src/</code> first to generate the model."
            )
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    context = {
        "brand_options":        _make_options(BRANDS,        fd.get("brand", "")),
        "fuel_options":         _make_options(FUEL_TYPES,    fd.get("fuel_type", "")),
        "tx_options":           _make_options(TRANSMISSIONS, fd.get("transmission", "")),
        "accident_no_sel":      fd.get("accident", "0") == "0",
        "accident_yes_sel":     fd.get("accident") == "1",
        "form_data":            fd,
        "result":               result,
        "error":                error,
    }
    return render(request, "predictor/index.html", context)
