import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from .data_cleaning import get_feature_target_split, load_and_clean_data, split_data
except ImportError:
    from data_cleaning import get_feature_target_split, load_and_clean_data, split_data


def build_pipelines():
    """Build model pipelines with selective scaling."""
    num_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
    cat_cols = ["origin"]

    # Unscaled transformer: passes numeric features through, encodes origin
    prep_unscaled = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(categories=[[1, 2, 3]], drop="first"), cat_cols),
        ]
    )

    # Scaled transformer: standardizes numeric features only, encodes origin
    prep_scaled = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(categories=[[1, 2, 3]], drop="first"), cat_cols),
        ]
    )

    # 4 regression models: Linear Regression & Random Forest unscaled, KNN & SVR scaled
    pipelines = {
        "Linear Regression": Pipeline(
            [("prep", prep_unscaled), ("model", LinearRegression())]
        ),
        "KNeighborsRegressor": Pipeline(
            [
                ("prep", prep_scaled),
                ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [("prep", prep_scaled), ("model", SVR(C=25.0, epsilon=0.5))]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", prep_unscaled),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        ),
    }

    return pipelines


def train_and_evaluate(data_path: str = None, project_root: str = None):
    """Train models, evaluate on 5-fold CV & held-out test set, and save artifacts."""
    if project_root is None:
        # Default to parent directory of src/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    df = load_and_clean_data(data_path)
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 5-fold cross validation with fixed seed for reproducibility
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipelines = build_pipelines()

    metrics_list = []
    test_predictions_df = X_test.copy()
    test_predictions_df["Actual MPG"] = y_test.values

    models_dir = os.path.join(project_root, "models")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    for name, pipe in pipelines.items():
        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        cv_r2_mean = float(cv_scores.mean())
        cv_r2_std = float(cv_scores.std())

        # Fit model on full training set
        pipe.fit(X_train, y_train)

        # Predict on held-out test set
        preds = pipe.predict(X_test)
        test_predictions_df[f"Predicted_{name}"] = preds

        # Evaluation metrics on test set
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        test_r2 = float(r2_score(y_test, preds))

        metrics_list.append(
            {
                "Model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Test R²": round(test_r2, 4),
                "CV R² (Mean)": round(cv_r2_mean, 4),
                "CV R² (Std)": round(cv_r2_std, 4),
            }
        )

        # Save serialized model pipeline
        clean_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(models_dir, clean_filename))

    metrics_df = pd.DataFrame(metrics_list)

    # Save metrics summary and test predictions
    metrics_json_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_list, f, indent=4)

    metrics_csv_path = os.path.join(models_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    predictions_csv_path = os.path.join(data_dir, "test_predictions.csv")
    test_predictions_df.to_csv(predictions_csv_path, index=False)

    print("--- Model Training & Evaluation Summary ---")
    print(metrics_df.to_string(index=False))
    print(f"\nArtifacts successfully saved to:\n  - {models_dir}\n  - {data_dir}")

    return metrics_df, test_predictions_df


if __name__ == "__main__":
    train_and_evaluate()
