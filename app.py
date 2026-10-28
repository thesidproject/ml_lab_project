import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import KFold, train_test_split

# Setup paths and import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_cleaning import clean_missing_values, find_dataset_path
from kneighbors_regressor.model import train_and_evaluate as train_knn
from linear_regression.model import train_and_evaluate as train_lr
from random_forest.model import train_and_evaluate as train_rf
from support_vector_machine.model import train_and_evaluate as train_svr

# Page configuration
st.set_page_config(
    page_title="Auto MPG Fuel Efficiency",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS: Reduce top padding, style tab buttons
st.markdown(
    """
    <style>
        /* Cut empty space at top */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }

        /* Typography */
        html, body, [class*="css"] {
            font-size: 18px !important;
        }
        h1 { font-size: 2.1rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
        h2 { font-size: 1.6rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.3rem !important; font-weight: 600 !important; }

        /* Distinct Tab Cards */
        div[data-baseweb="tab-list"] {
            gap: 12px !important;
            padding-bottom: 10px !important;
            border-bottom: 2px solid rgba(128, 128, 128, 0.2) !important;
        }
        button[data-baseweb="tab"] {
            background-color: rgba(128, 128, 128, 0.12) !important;
            border: 1px solid rgba(128, 128, 128, 0.3) !important;
            border-radius: 8px !important;
            padding: 8px 22px !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: inherit !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: inherit !important;
        }
        button[data-baseweb="tab"]:hover {
            background-color: rgba(128, 128, 128, 0.22) !important;
            border-color: #2b6cb0 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #2b6cb0 !important;
            border-color: #2b6cb0 !important;
            box-shadow: 0 2px 8px rgba(43, 108, 176, 0.35) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] p,
        button[data-baseweb="tab"][aria-selected="true"] span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        div[data-baseweb="tab-highlight"],
        div[data-baseweb="tab-border"] {
            display: none !important;
            background-color: transparent !important;
        }

        /* Metric card values */
        [data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }

        /* Primary Button */
        .stButton>button, .stFormSubmitButton>button {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Auto MPG Predictive Analytics")

# Load raw dataset
@st.cache_data
def get_raw_data():
    raw_path = find_dataset_path()
    return pd.read_csv(raw_path)

raw_df = get_raw_data()


def cleanup_all_artifacts():
    """Remove generated model binaries and cache files."""
    import glob
    import shutil

    for folder in ["saved_models", "models"]:
        for pattern in ["*.joblib", "**/*.joblib"]:
            for f in glob.glob(os.path.join(BASE_DIR, folder, pattern), recursive=True):
                try:
                    os.remove(f)
                except OSError:
                    pass

    for path in [
        os.path.join(BASE_DIR, "data", "test_predictions.csv"),
        os.path.join(BASE_DIR, "saved_models", "evaluation_metrics.json"),
        os.path.join(BASE_DIR, "saved_models", "metrics_summary.csv"),
    ]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    for root, dirs, _ in os.walk(BASE_DIR):
        for d in dirs:
            if d == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(root, d))
                except OSError:
                    pass


# Sidebar: Minimal maintenance controls
with st.sidebar:
    st.subheader("Controls")
    if st.button("Reset Artifacts", use_container_width=True):
        cleanup_all_artifacts()
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Clean data automatically
cleaned_df = clean_missing_values(raw_df)

# Initialize Session State
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "trained_pipelines" not in st.session_state:
    st.session_state.trained_pipelines = {}
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "test_predictions_df" not in st.session_state:
    st.session_state.test_predictions_df = None


def try_load_existing_artifacts():
    """Load existing model artifacts if available on disk."""
    metrics_path = os.path.join(BASE_DIR, "saved_models", "metrics_summary.csv")
    predictions_path = os.path.join(BASE_DIR, "data", "test_predictions.csv")

    if os.path.exists(metrics_path) and os.path.exists(predictions_path):
        try:
            import joblib

            loaded = {}
            configs = [
                ("Linear Regression", "saved_models/linear_regression/linear_regression.joblib"),
                ("KNeighborsRegressor", "saved_models/kneighbors_regressor/kneighbors_regressor.joblib"),
                ("Support Vector Machine", "saved_models/support_vector_machine/support_vector_machine.joblib"),
                ("Random Forest", "saved_models/random_forest/random_forest.joblib"),
            ]
            for name, rel_path in configs:
                full_path = os.path.join(BASE_DIR, rel_path)
                if os.path.exists(full_path):
                    loaded[name] = joblib.load(full_path)

            if loaded and not st.session_state.models_trained:
                st.session_state.trained_pipelines = loaded
                st.session_state.metrics_df = pd.read_csv(metrics_path)
                st.session_state.test_predictions_df = pd.read_csv(predictions_path)
                st.session_state.models_trained = True
        except Exception:
            pass


try_load_existing_artifacts()

# Short, distinct tab names without numbers or emojis
tab_data, tab_train, tab_chart, tab_results, tab_predict = st.tabs(
    ["Data", "Train", "Chart", "Results", "Predict"]
)

# ==========================================
# TAB 1: DATA
# ==========================================
with tab_data:
    st.header("Dataset Inspection & Cleaning")

    st.markdown("### Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    missing_count = int((raw_df["horsepower"] == "?").sum())

    if "show_missing_inspection" not in st.session_state:
        st.session_state.show_missing_inspection = False

    st.markdown("### Missing Value Inspection")
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Inspect Missing Values", use_container_width=True):
            st.session_state.show_missing_inspection = not st.session_state.show_missing_inspection

    if st.session_state.show_missing_inspection:
        st.warning(
            f"Found {missing_count} vehicles containing missing horsepower values (marked as '?'). "
            "These are automatically imputed using the median horsepower of their cylinder group."
        )

        st.markdown("### Vehicles with Missing Horsepower")
        missing_rows = raw_df[raw_df["horsepower"] == "?"][
            ["car name", "cylinders", "displacement", "horsepower", "weight"]
        ]
        st.dataframe(missing_rows, use_container_width=True, hide_index=True)

        st.markdown("### Cleaning & Imputation Function (`src/data_cleaning.py`)")
        st.code(
            """def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["horsepower"] = pd.to_numeric(cleaned["horsepower"].replace("?", np.nan))
    # Impute missing horsepower using cylinder-group median
    cleaned["horsepower"] = cleaned["horsepower"].fillna(
        cleaned.groupby("cylinders")["horsepower"].transform("median")
    )
    # Global median fallback if any remain
    cleaned["horsepower"] = cleaned["horsepower"].fillna(cleaned["horsepower"].median())
    # Place 'car name' as the first column for clean preview
    if "car name" in cleaned.columns:
        cols = ["car name"] + [c for c in cleaned.columns if c != "car name"]
        cleaned = cleaned[cols]
    return cleaned""",
            language="python",
        )

# ==========================================
# TAB 2: TRAIN
# ==========================================
with tab_train:
    st.header("Data Splitting & Model Training")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        test_ratio = st.slider("Test Split (%)", min_value=10, max_value=40, value=20, step=5)
    with col_s2:
        random_seed = st.number_input(
            "Random State (Seed)",
            min_value=0,
            max_value=999,
            value=42,
            step=1,
            help="Fixed seed ensuring deterministic splits and identical reproducible results.",
        )
    with col_s3:
        n_total = len(cleaned_df)
        n_test = int(n_total * (test_ratio / 100.0))
        n_train = n_total - n_test
        st.metric("Train Samples", n_train)
    with col_s4:
        st.metric("Test Samples", n_test)

    # Feature and Target Split
    features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
    X = cleaned_df[features].copy()
    y = cleaned_df["mpg"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_ratio / 100.0), random_state=random_seed
    )

    selected_models = st.multiselect(
        "Select Models to Train",
        options=[
            "Linear Regression",
            "KNeighborsRegressor",
            "Support Vector Machine",
            "Random Forest",
        ],
        default=[
            "Linear Regression",
            "KNeighborsRegressor",
            "Support Vector Machine",
            "Random Forest",
        ],
    )

    if st.button("Train Selected Models", type="primary", use_container_width=True):
        runner_map = {
            "Linear Regression": (train_lr, "linear_regression"),
            "KNeighborsRegressor": (train_knn, "kneighbors_regressor"),
            "Support Vector Machine": (train_svr, "support_vector_machine"),
            "Random Forest": (train_rf, "random_forest"),
        }

        selected_runners = [(name, runner_map[name][0], runner_map[name][1]) for name in selected_models if name in runner_map]

        if not selected_runners:
            st.warning("Please select at least one model to train.")
        else:
            with st.spinner("Training models with 5-fold cross-validation..."):
                kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
                trained_pipes = {}
                metrics_list = []
                test_preds_df = X_test.copy()
                test_preds_df["Actual MPG"] = y_test.values

                for name, runner_fn, folder_name in selected_runners:
                    save_dir = os.path.join(BASE_DIR, "saved_models", folder_name)
                    pipe, metrics, preds = runner_fn(
                        X_train, y_train, X_test, y_test, kf=kf, save_dir=save_dir
                    )

                    trained_pipes[name] = pipe
                    metrics_list.append(metrics)
                    test_preds_df[f"Predicted_{name}"] = preds

                st.session_state.trained_pipelines = trained_pipes
                st.session_state.metrics_df = pd.DataFrame(metrics_list)
                st.session_state.test_predictions_df = test_preds_df
                st.session_state.models_trained = True

                # Persist summaries to disk
                st.session_state.metrics_df.to_csv(
                    os.path.join(BASE_DIR, "saved_models", "metrics_summary.csv"), index=False
                )
                test_preds_df.to_csv(
                    os.path.join(BASE_DIR, "data", "test_predictions.csv"), index=False
                )

            st.success(f"Training complete. {len(selected_runners)} models saved.")

# ==========================================
# TAB 3: CHART
# ==========================================
with tab_chart:
    st.header("Actual vs. Predicted MPG")

    if (
        st.session_state.test_predictions_df is not None
        and len(st.session_state.trained_pipelines) > 0
    ):
        chart_col, _ = st.columns([1, 2])
        with chart_col:
            chart_model = st.selectbox(
                "Select Model",
                options=list(st.session_state.trained_pipelines.keys()),
            )

        pred_col = f"Predicted_{chart_model}"
        if pred_col not in st.session_state.test_predictions_df.columns:
            st.warning(f"Predictions for '{chart_model}' were not found in current test predictions. Please retrain.")
        else:
            y_actual = st.session_state.test_predictions_df["Actual MPG"]
            y_predicted = st.session_state.test_predictions_df[pred_col]
            abs_errors = np.abs(y_actual - y_predicted)

            min_val = min(y_actual.min(), y_predicted.min()) - 2
            max_val = max(y_actual.max(), y_predicted.max()) + 2

            fig = go.Figure()

            # Perfect prediction dashed line (y = x)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ideal (y = x)",
                    line=dict(color="#888888", dash="dash", width=2),
                    showlegend=False,
                )
            )

            # Actual vs Predicted points
            fig.add_trace(
                go.Scatter(
                    x=y_actual,
                    y=y_predicted,
                    mode="markers",
                    name=chart_model,
                    marker=dict(
                        size=10,
                        color=abs_errors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Absolute Error"),
                        opacity=0.85,
                    ),
                    text=[
                        f"Actual: {a:.1f} MPG<br>Predicted: {p:.1f} MPG<br>Error: {e:.2f} MPG"
                        for a, p, e in zip(y_actual, y_predicted, abs_errors)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            fig.update_layout(
                title=f"Actual vs. Predicted — {chart_model}",
                xaxis_title="Actual MPG",
                yaxis_title="Predicted MPG",
                xaxis=dict(range=[min_val, max_val]),
                yaxis=dict(range=[min_val, max_val]),
                height=440,
                margin=dict(l=40, r=40, t=45, b=55),
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models trained yet. Go to the 'Train' tab and click Train Selected Models.")

# ==========================================
# TAB 4: RESULTS (PERFORMANCE)
# ==========================================
with tab_results:
    st.header("Model Performance")

    if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
        m_df = st.session_state.metrics_df.copy()
        if "CV R² (Mean)" in m_df.columns and "CV R² (Std)" in m_df.columns:
            m_df["CV R²"] = (
                m_df["CV R² (Mean)"].map("{:.4f}".format)
                + " ± "
                + m_df["CV R² (Std)"].map("{:.4f}".format)
            )
        elif "CV R²" not in m_df.columns:
            m_df["CV R²"] = "N/A"

        if "MSE" not in m_df.columns and "RMSE" in m_df.columns:
            m_df["MSE"] = (m_df["RMSE"] ** 2).round(4)

        all_cols = ["Model", "MAE", "MSE", "RMSE", "Test R²", "CV R²"]
        show_cols = [c for c in all_cols if c in m_df.columns]
        display_table = m_df[show_cols]
        if "Test R²" in display_table.columns:
            display_table = display_table.sort_values(by="Test R²", ascending=False)

        best_row = display_table.iloc[0]

        # Bordered container for the winning model
        with st.container(border=True):
            st.markdown(f"### Top Model: **{best_row['Model']}**")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Test R²", f"{best_row['Test R²']:.4f}")
            b2.metric("Test MSE", f"{best_row['MSE']:.4f}")
            b3.metric("Test RMSE", f"{best_row['RMSE']:.4f}")
            b4.metric("Test MAE", f"{best_row['MAE']:.4f}")
            b5.metric("CV R²", best_row["CV R²"])

        st.caption("Metrics guide: Higher R² is better; lower error (MAE, MSE, RMSE) is better.")

        # Highlight best values: Max for R², Min for errors
        styled_df = (
            display_table.style.format(
                {
                    "MAE": "{:.4f}",
                    "MSE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "Test R²": "{:.4f}",
                }
            )
            .highlight_max(subset=["Test R²"], color="#c6f6d5")
            .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#c6f6d5")
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No models trained yet. Go to the 'Train' tab and click Train Selected Models.")

# ==========================================
# TAB 5: PREDICT
# ==========================================
with tab_predict:
    st.header("Vehicle Fuel Efficiency Predictor")

    if len(st.session_state.trained_pipelines) > 0:
        with st.form("vehicle_prediction_form"):
            # Model selector first in its own full row
            in_model_choice = st.selectbox(
                "Prediction Model",
                options=list(st.session_state.trained_pipelines.keys()),
            )

            row1_c1, row1_c2, row1_c3 = st.columns(3)
            with row1_c1:
                in_cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
            with row1_c2:
                in_displacement = st.number_input(
                    "Displacement (cu. in.)", min_value=60, max_value=500, value=150, step=1
                )
            with row1_c3:
                in_horsepower = st.number_input(
                    "Horsepower", min_value=40, max_value=300, value=95, step=1
                )

            row2_c1, row2_c2, row2_c3 = st.columns(3)
            with row2_c1:
                in_weight = st.number_input(
                    "Weight (lbs)", min_value=1500, max_value=5500, value=2800, step=10
                )
            with row2_c2:
                in_acceleration = st.number_input(
                    "Acceleration (0-60 mph, sec)",
                    min_value=5.0,
                    max_value=30.0,
                    value=15.5,
                    step=0.1,
                    format="%.1f",
                )
            with row2_c3:
                in_year = st.slider(
                    "Model Year (1970 - 1982)", min_value=70, max_value=82, value=76, step=1
                )

            in_origin_label = st.selectbox(
                "Origin", options=["USA (1)", "Europe (2)", "Japan (3)"], index=0
            )
            in_origin_map = {"USA (1)": 1, "Europe (2)": 2, "Japan (3)": 3}
            in_origin_val = in_origin_map[in_origin_label]

            submit_pred = st.form_submit_button(
                "Predict Fuel Efficiency", type="primary", use_container_width=True
            )

        if submit_pred:
            input_row = pd.DataFrame(
                [
                    {
                        "cylinders": in_cylinders,
                        "displacement": in_displacement,
                        "horsepower": in_horsepower,
                        "weight": in_weight,
                        "acceleration": in_acceleration,
                        "model year": in_year,
                        "origin": in_origin_val,
                    }
                ]
            )

            if in_model_choice in st.session_state.trained_pipelines:
                chosen_pipe = st.session_state.trained_pipelines[in_model_choice]
                try:
                    raw_pred = chosen_pipe.predict(input_row)[0]
                    est_mpg = max(1.0, float(raw_pred))
                    est_kmpl = est_mpg * 0.425144

                    # Large, prominent result container below button
                    with st.container(border=True):
                        r1, r2 = st.columns(2)
                        r1.metric("Predicted Fuel Efficiency", f"{est_mpg:.1f} MPG")
                        r2.metric("Metric Equivalent", f"{est_kmpl:.1f} km/L")
                except Exception as err:
                    st.error(f"Prediction failed with error: {err}")
            else:
                st.warning(f"Selected model '{in_model_choice}' is not available. Please train it in the 'Train' tab.")
    else:
        st.info("Please train the models first in the 'Train' tab.")
