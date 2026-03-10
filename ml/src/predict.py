"""
predict.py — Load the saved pipeline and predict. 
One call: pipeline.predict(X) handles all encoding, scaling, and inference.
"""
import pandas as pd
from pipeline import load_pipeline


def predict(input_dict: dict) -> float:
    """
    Predict price from raw feature dictionary.
    The pipeline handles encoding + scaling + model internally.

    Example:
    {
        'brand': 'Toyota', 'milage': 45000.0, 'fuel_type': 'Gasoline',
        'transmission': 'automatic', 'accident': 0,
        'tx_speed': 6, 'horse_power': 180.0,
        'engine_litres': 2.5, 'car_age': 8
    }
    """
    pipeline = load_pipeline()
    X = pd.DataFrame([input_dict])
    price = pipeline.predict(X)[0]
    print(f"Predicted Price: ${price:,.2f}")
    return price


if __name__ == "__main__":
    sample = {
        "brand"        : "Toyota",
        "milage"       : 45000.0,
        "fuel_type"    : "Gasoline",
        "transmission" : "automatic",
        "accident"     : 0,
        "tx_speed"     : 6,
        "horse_power"  : 180.0,
        "engine_litres": 2.5,
        "car_age"      : 8,
    }
    predict(sample)
