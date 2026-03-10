"""
train.py — Train the pipeline, evaluate, and save.
"""
from preprocessing import get_splits
from pipeline import build_pipeline, save_pipeline
from evaluate import evaluate_model, plot_predictions, plot_feature_importance


if __name__ == "__main__":
    # 1. Load clean data and split
    X_train, X_test, y_train, y_test = get_splits()

    # 2. Build and fit the full pipeline (preprocessing + model in one step)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="XGBoost Pipeline (Baseline)")

    # 4. Plots
    plot_predictions(y_test, y_pred, model_name="XGBoost Pipeline (Baseline)")

    xgb_model = pipeline.named_steps["model"]
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist()
    )
    plot_feature_importance(xgb_model, feature_names, top_n=15)

    # 5. Save the entire pipeline
    save_pipeline(pipeline)
