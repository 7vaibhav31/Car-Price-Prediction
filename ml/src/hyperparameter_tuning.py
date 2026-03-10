"""
hyperparameter_tuning.py — Tune XGBoost inside the Pipeline using RandomizedSearchCV.
All param names are prefixed with 'model__' to target the XGBRegressor step.
"""
from sklearn.model_selection import RandomizedSearchCV, KFold

from preprocessing import get_splits
from pipeline import build_pipeline, save_pipeline
from evaluate import evaluate_model

# ── Search space (prefix 'model__' targets XGBRegressor inside the pipeline)
PARAM_DIST = {
    "model__n_estimators"    : [100, 200, 300, 500],
    "model__max_depth"       : [3, 4, 5, 6, 7, 8],
    "model__learning_rate"   : [0.01, 0.05, 0.1, 0.15, 0.2],
    "model__subsample"       : [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__gamma"           : [0, 0.1, 0.2, 0.3],
    "model__reg_alpha"       : [0, 0.01, 0.1, 1],
    "model__reg_lambda"      : [1, 1.5, 2, 3],
}


if __name__ == "__main__":
    # 1. Clean data + split
    X_train, X_test, y_train, y_test = get_splits()

    # 2. Build base pipeline
    pipeline = build_pipeline()

    # 3. RandomizedSearchCV wraps the WHOLE pipeline
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DIST,
        n_iter=50,
        scoring="r2",
        cv=kf,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Starting hyperparameter search...")
    search.fit(X_train, y_train)

    print(f"\nBest R² (CV) : {search.best_score_:.4f}")
    print(f"Best Params  : {search.best_params_}")

    # 4. Evaluate best pipeline on test set
    y_pred = search.best_estimator_.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="XGBoost Pipeline (Tuned)")

    # 5. Save the best pipeline (includes preprocessor + tuned model)
    save_pipeline(search.best_estimator_)
