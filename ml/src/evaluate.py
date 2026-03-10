"""
evaluate.py — Evaluation metrics and reporting for regression models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Print and return regression evaluation metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'='*40}")
    return metrics


def plot_predictions(y_true, y_pred, model_name: str = "Model"):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, color="steelblue", edgecolors="none")
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"{model_name} — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """Plot top-N feature importances (for tree-based models)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="steelblue"
    )
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("evaluate.py — import and use evaluate_model(), plot_predictions(), plot_feature_importance()")
