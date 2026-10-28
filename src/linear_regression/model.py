import os
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Add parent directory to sys.path for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_cleaning import get_feature_target_split, load_and_clean_data, split_data


def build_pipeline():
    """Build Linear Regression pipeline (unscaled features for direct interpretability)."""
    num_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
    cat_cols = ["origin"]

    # Passthrough numeric features without scaling
    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(categories=[[1, 2, 3]], drop="first"), cat_cols),
        ]
    )

    pipeline = Pipeline([("prep", prep), ("model", LinearRegression())])
    return pipeline


def train_and_evaluate(X_train, y_train, X_test, y_test, kf=None, save_dir=None):
    """Train Linear Regression, compute 5-fold CV & test metrics, and optionally save model."""
    if kf is None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = build_pipeline()

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
        "Model": "Linear Regression",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Test R²": round(test_r2, 4),
        "CV R² (Mean)": round(cv_mean, 4),
        "CV R² (Std)": round(cv_std, 4),
    }

    # Optionally save model artifact
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(save_dir, "linear_regression.joblib"))

    return pipeline, metrics, preds


if __name__ == "__main__":
    df = load_and_clean_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_dir = os.path.join(current_dir, "..", "..", "saved_models", "linear_regression")
    pipe, metrics, _ = train_and_evaluate(X_train, y_train, X_test, y_test, save_dir=model_dir)
    print("Linear Regression Metrics:", metrics)
