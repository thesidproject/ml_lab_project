import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Add src to sys.path for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_cleaning import get_feature_target_split, load_and_clean_data, split_data
from evaluation import evaluate_pipeline


def build_pipeline():
    """Build Linear Regression pipeline (unscaled features for direct interpretability)."""
    num_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
    cat_cols = ["origin"]

    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(categories=[[1, 2, 3]], drop="first"), cat_cols),
        ]
    )

    return Pipeline([("prep", prep), ("model", LinearRegression())])


def train_and_evaluate(X_train, y_train, X_test, y_test, kf=None, save_dir=None):
    """Train Linear Regression and evaluate using shared evaluation engine."""
    pipeline = build_pipeline()
    return evaluate_pipeline(pipeline, "Linear Regression", X_train, y_train, X_test, y_test, kf, save_dir)


if __name__ == "__main__":
    df = load_and_clean_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_dir = os.path.join(src_dir, "..", "saved_models", "linear_regression")
    pipe, metrics, _ = train_and_evaluate(X_train, y_train, X_test, y_test, save_dir=save_dir)
    print("Linear Regression Metrics:", metrics)
