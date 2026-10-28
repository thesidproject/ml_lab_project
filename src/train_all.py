import json
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold

# Ensure src/ is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_cleaning import get_feature_target_split, load_and_clean_data, split_data
from kneighbors_regressor import train_and_evaluate as train_knn
from linear_regression import train_and_evaluate as train_lr
from random_forest import train_and_evaluate as train_rf
from support_vector_machine import train_and_evaluate as train_svr


def train_all_models(data_path: str = None, test_size: float = 0.2, random_state: int = 42, project_root: str = None):
    """Orchestrate training of all 4 individual model modules and collect metrics."""
    if project_root is None:
        project_root = os.path.abspath(os.path.join(current_dir, ".."))

    df = load_and_clean_data(data_path)
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    saved_models_dir = os.path.join(project_root, "saved_models")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    test_predictions_df = X_test.copy()
    test_predictions_df["Actual MPG"] = y_test.values

    runners = [
        ("Linear Regression", train_lr, "linear_regression"),
        ("KNeighborsRegressor", train_knn, "kneighbors_regressor"),
        ("Support Vector Machine", train_svr, "support_vector_machine"),
        ("Random Forest", train_rf, "random_forest"),
    ]

    trained_pipelines = {}
    metrics_list = []

    for name, runner_fn, folder_name in runners:
        sub_model_dir = os.path.join(saved_models_dir, folder_name)
        pipe, metrics, preds = runner_fn(
            X_train, y_train, X_test, y_test, kf=kf, save_dir=sub_model_dir
        )

        # Also save top-level copy in saved_models/ for convenience
        import joblib
        joblib.dump(pipe, os.path.join(saved_models_dir, f"{folder_name}.joblib"))

        trained_pipelines[name] = pipe
        metrics_list.append(metrics)
        test_predictions_df[f"Predicted_{name}"] = preds

    metrics_df = pd.DataFrame(metrics_list)

    # Save summaries
    metrics_df.to_csv(os.path.join(saved_models_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(saved_models_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_list, f, indent=4)
    test_predictions_df.to_csv(
        os.path.join(data_dir, "test_predictions.csv"), index=False
    )

    return trained_pipelines, metrics_df, test_predictions_df


if __name__ == "__main__":
    _, metrics_df, _ = train_all_models()
    print("--- All Models Successfully Trained ---")
    print(metrics_df.to_string(index=False))
