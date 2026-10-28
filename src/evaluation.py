import os
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


def evaluate_pipeline(pipeline, model_name: str, X_train, y_train, X_test, y_test, kf=None, save_dir: str = None):
    """Common evaluation engine: computes 5-fold CV, fits pipeline, evaluates on test set, and saves artifact."""
    if kf is None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Predict on held-out test data
    preds = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    test_r2 = float(r2_score(y_test, preds))

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Test R²": round(test_r2, 4),
        "CV R² (Mean)": round(cv_mean, 4),
        "CV R² (Std)": round(cv_std, 4),
    }

    # Save model artifact if destination directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(save_dir, filename))

    return pipeline, metrics, preds
