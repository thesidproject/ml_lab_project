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
    page_title="Auto MPG Fuel Efficiency Studio",
    page_icon="🚗",
    layout="wide",
)

# Custom CSS for significantly larger, prominent fonts
st.markdown(
    """
    <style>
        /* Base typography */
        html, body, [class*="css"] {
            font-size: 18px !important;
        }
        /* Prominent Headings */
        h1 { font-size: 2.3rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.8rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.4rem !important; font-weight: 600 !important; }
        p, div { font-size: 1.1rem !important; }

        /* Tab Bar Layout & Visual Separation */
        div[data-baseweb="tab-list"] {
            gap: 12px !important;
            padding-bottom: 8px !important;
            border-bottom: 2px solid rgba(128, 128, 128, 0.2) !important;
        }

        /* Distinct Tab Cards */
        button[data-baseweb="tab"] {
            background-color: rgba(128, 128, 128, 0.08) !important;
            border: 1px solid rgba(128, 128, 128, 0.25) !important;
            border-radius: 8px !important;
            padding: 10px 22px !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            color: inherit !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Hover State */
        button[data-baseweb="tab"]:hover {
            background-color: rgba(128, 128, 128, 0.18) !important;
            border-color: rgba(128, 128, 128, 0.5) !important;
        }

        /* Active / Selected Tab Card */
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1f6feb !important;
            color: #ffffff !important;
            border-color: #1f6feb !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25) !important;
        }

        /* Remove default thin bottom highlight line */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Larger Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }

        /* Larger Form & Button Controls */
        .stButton>button {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
        }
        label {
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Auto MPG Predictive Analytics Studio")

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
        for f in glob.glob(os.path.join(BASE_DIR, folder, "**", "*.joblib"), recursive=True):
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


# Sidebar: Minimal maintenance controls only
with st.sidebar:
    st.header("Project Controls")
    if st.button("🧹 Clean All Artifacts & Reset", use_container_width=True):
        cleanup_all_artifacts()
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Automatically clean data using the standard pipeline
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
    """Load existing model artifacts if available on disk and not yet in session."""
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


# Try loading pre-existing models on startup
try_load_existing_artifacts()

# Tabbed Interface (Clean & Distinct)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Data & Missing Values",
        "2. Split & Train Models",
        "3. Model Performance",
        "4. Actual vs. Predicted Chart",
        "5. Vehicle Fuel Predictor",
    ]
)

# ==========================================
# TAB 1: DATA & MISSING VALUES
# ==========================================
with tab1:
    st.header("Data Inspection & Missing Value Analysis")
    st.markdown("Inspect raw data, identify missing values, and review the cleaning function.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Raw Dataset Preview")
        st.dataframe(raw_df.head(8), use_container_width=True)

    with c2:
        st.markdown("### Missing Value Check")
        missing_count = int((raw_df["horsepower"] == "?").sum())
        st.metric("Missing Values Detected", f"{missing_count} rows in 'horsepower'")

        if st.button("🔍 Inspect Missing Values", use_container_width=True):
            st.session_state.show_missing = True

    if st.session_state.get("show_missing", False):
        st.markdown("### Vehicles with Missing Horsepower ('?')")
        missing_rows = raw_df[raw_df["horsepower"] == "?"][
            ["car name", "cylinders", "displacement", "horsepower", "weight"]
        ]
        st.dataframe(missing_rows, use_container_width=True)

        st.markdown("### Cleaning Function Applied (`data_cleaning.py`)")
        st.code(
            """def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["horsepower"] = pd.to_numeric(cleaned["horsepower"].replace("?", np.nan))
    # Impute missing horsepower using cylinder-group median
    cleaned["horsepower"] = cleaned["horsepower"].fillna(
        cleaned.groupby("cylinders")["horsepower"].transform("median")
    )
    return cleaned""",
            language="python",
        )

        st.success(
            f"✅ Cleaned Dataset Ready: All missing values resolved. 0 missing values remain across all {len(cleaned_df)} rows."
        )

# ==========================================
# TAB 2: SPLIT & TRAIN MODELS
# ==========================================
with tab2:
    st.header("Train / Test Splitting & Model Training")
    st.markdown("Configure the data split, select models, and trigger training with 5-fold cross-validation.")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        test_ratio = st.slider("Test Set Split Ratio (%)", min_value=10, max_value=40, value=20, step=5)
    with s_col2:
        random_seed = st.number_input(
            "Random State (Seed)",
            min_value=0,
            max_value=999,
            value=42,
            step=1,
            help="Fixed seed ensuring deterministic splits and identical reproducible results.",
        )
    with s_col3:
        n_total = len(cleaned_df)
        n_test = int(n_total * (test_ratio / 100.0))
        n_train = n_total - n_test
        st.metric("Dataset Partition", f"Train: {n_train} | Test: {n_test}")

    # Feature and Target Split
    features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
    X = cleaned_df[features].copy()
    y = cleaned_df["mpg"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_ratio / 100.0), random_state=random_seed
    )

    st.markdown("### Select Models to Train")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        use_lr = st.checkbox("Linear Regression (Unscaled)", value=True)
    with m_col2:
        use_knn = st.checkbox("KNeighborsRegressor (StandardScaler)", value=True)
    with m_col3:
        use_svr = st.checkbox("Support Vector Machine (StandardScaler)", value=True)
    with m_col4:
        use_rf = st.checkbox("Random Forest (Unscaled)", value=True)

    if st.button("🚀 Train Selected Models with 5-Fold CV", type="primary", use_container_width=True):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        selected_runners = []
        if use_lr:
            selected_runners.append(("Linear Regression", train_lr, "linear_regression"))
        if use_knn:
            selected_runners.append(("KNeighborsRegressor", train_knn, "kneighbors_regressor"))
        if use_svr:
            selected_runners.append(("Support Vector Machine", train_svr, "support_vector_machine"))
        if use_rf:
            selected_runners.append(("Random Forest", train_rf, "random_forest"))

        if not selected_runners:
            st.warning("Please select at least one model to train.")
        else:
            with st.spinner("Training models with 5-fold cross-validation and saving to saved_models/..."):
                trained_pipes = {}
                metrics_list = []
                test_preds_df = X_test.copy()
                test_preds_df["Actual MPG"] = y_test.values

                for name, runner_fn, folder_name in selected_runners:
                    # Save trained model to project/saved_models/<model_name>/
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

                # Persist evaluation summaries to disk
                st.session_state.metrics_df.to_csv(
                    os.path.join(BASE_DIR, "saved_models", "metrics_summary.csv"), index=False
                )
                test_preds_df.to_csv(
                    os.path.join(BASE_DIR, "data", "test_predictions.csv"), index=False
                )

            st.success(
                f"✅ Training completed! Trained models saved directly to 'project/saved_models/'."
            )

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Model Performance Analytics")
    if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
        m_df = st.session_state.metrics_df.copy()
        m_df["CV R²"] = (
            m_df["CV R² (Mean)"].map("{:.4f}".format)
            + " ± "
            + m_df["CV R² (Std)"].map("{:.4f}".format)
        )
        display_table = m_df[
            ["Model", "MAE", "MSE", "RMSE", "Test R²", "CV R²"]
        ].sort_values(by="Test R²", ascending=False)

        st.dataframe(
            display_table.style.format(
                {
                    "MAE": "{:.4f}",
                    "MSE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "Test R²": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        best_row = display_table.iloc[0]
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Top Model (by Test R²)", best_row["Model"])
        bc2.metric("Test R²", f"{best_row['Test R²']:.4f}")
        bc3.metric("Test MSE", f"{best_row['MSE']:.4f}")
        bc4.metric("Test RMSE", f"{best_row['RMSE']:.4f}")
        bc5.metric("Test MAE", f"{best_row['MAE']:.4f}")
    else:
        st.info("No models trained yet. Go to 'Tab 2: Split & Train Models' and click Train.")

# ==========================================
# TAB 4: ACTUAL VS. PREDICTED CHART
# ==========================================
with tab4:
    st.header("Interactive Actual vs. Predicted Parity Chart")
    if (
        st.session_state.test_predictions_df is not None
        and len(st.session_state.trained_pipelines) > 0
    ):
        chart_model = st.selectbox(
            "Select Model to Inspect:",
            options=list(st.session_state.trained_pipelines.keys()),
        )

        pred_col = f"Predicted_{chart_model}"
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
                name="Perfect Prediction (y = x)",
                line=dict(color="#888888", dash="dash", width=2),
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
            )
        )

        fig.update_layout(
            title=f"Actual vs. Predicted Fuel Efficiency — {chart_model}",
            xaxis_title="Actual MPG",
            yaxis_title="Predicted MPG",
            xaxis=dict(range=[min_val, max_val]),
            yaxis=dict(range=[min_val, max_val]),
            height=580,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models trained yet. Go to 'Tab 2: Split & Train Models' and click Train.")

# ==========================================
# TAB 5: VEHICLE FUEL PREDICTOR
# ==========================================
with tab5:
    st.header("Vehicle Fuel Efficiency Estimator (Inference)")
    st.markdown("Specify vehicle characteristics to predict estimated fuel efficiency:")

    if len(st.session_state.trained_pipelines) > 0:
        with st.form("vehicle_prediction_form"):
            row1_c1, row1_c2, row1_c3 = st.columns(3)
            with row1_c1:
                in_cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
            with row1_c2:
                in_displacement = st.number_input(
                    "Displacement (cu. in.)", min_value=60.0, max_value=500.0, value=150.0, step=5.0
                )
            with row1_c3:
                in_horsepower = st.number_input(
                    "Horsepower", min_value=40.0, max_value=300.0, value=95.0, step=1.0
                )

            row2_c1, row2_c2, row2_c3 = st.columns(3)
            with row2_c1:
                in_weight = st.number_input(
                    "Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2800.0, step=50.0
                )
            with row2_c2:
                in_acceleration = st.number_input(
                    "Acceleration (0-60 mph, sec)",
                    min_value=5.0,
                    max_value=30.0,
                    value=15.5,
                    step=0.5,
                )
            with row2_c3:
                in_year = st.slider(
                    "Model Year (1970 - 1982)", min_value=70, max_value=82, value=76
                )

            row3_c1, row3_c2 = st.columns([1, 2])
            with row3_c1:
                in_origin_label = st.selectbox(
                    "Origin", options=["USA (1)", "Europe (2)", "Japan (3)"], index=0
                )
                in_origin_map = {"USA (1)": 1, "Europe (2)": 2, "Japan (3)": 3}
                in_origin_val = in_origin_map[in_origin_label]

            with row3_c2:
                in_model_choice = st.selectbox(
                    "Select Prediction Model",
                    options=list(st.session_state.trained_pipelines.keys()),
                )

            submit_pred = st.form_submit_button(
                "Calculate Estimated Fuel Efficiency", use_container_width=True
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

            chosen_pipe = st.session_state.trained_pipelines[in_model_choice]
            est_mpg = max(1.0, float(chosen_pipe.predict(input_row)[0]))
            est_kmpl = est_mpg * 0.425144

            res_col1, res_col2 = st.columns(2)
            res_col1.success(f"### Predicted Fuel Efficiency: **{est_mpg:.2f} MPG**")
            res_col2.info(f"### Metric Equivalent: **{est_kmpl:.2f} km/L**")
    else:
        st.info("Please train the models first in 'Tab 2: Split & Train Models'.")
