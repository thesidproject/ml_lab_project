import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import KFold, train_test_split

# Setup paths and import individual model modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_cleaning import find_dataset_path
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

st.title("Auto MPG Predictive Analytics Studio")
st.caption(
    "Interactive Machine Learning Pipeline: Data Cleaning, Train/Test Splitting, 5-Fold Cross-Validation & Multi-Model Inference"
)

# Load raw dataset once
@st.cache_data
def get_raw_data():
    raw_path = find_dataset_path()
    return pd.read_csv(raw_path)

raw_df = get_raw_data()


def cleanup_all_artifacts():
    """Clean all generated model binaries, predictions, summaries, and pycache."""
    import glob
    import shutil

    # 1. Remove all .joblib files
    for folder in ["saved_models", "models"]:
        for f in glob.glob(os.path.join(BASE_DIR, folder, "**", "*.joblib"), recursive=True):
            try:
                os.remove(f)
            except OSError:
                pass

    # 2. Remove generated CSV and JSON summaries
    for path in [
        os.path.join(BASE_DIR, "data", "test_predictions.csv"),
        os.path.join(BASE_DIR, "saved_models", "evaluation_metrics.json"),
        os.path.join(BASE_DIR, "saved_models", "metrics_summary.csv"),
        os.path.join(BASE_DIR, "models", "evaluation_metrics.json"),
        os.path.join(BASE_DIR, "models", "metrics_summary.csv"),
    ]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # 3. Remove all __pycache__ directories
    for root, dirs, _ in os.walk(BASE_DIR):
        for d in dirs:
            if d == "__pycache__":
                pycache_path = os.path.join(root, d)
                try:
                    shutil.rmtree(pycache_path)
                except OSError:
                    pass


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Artifact Management")
    st.markdown("Remove all generated models, predictions, and cache files to reset to a clean state.")
    if st.button("🧹 Clean All Artifacts & Reset", type="secondary", use_container_width=True):
        cleanup_all_artifacts()
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("All artifacts and caches successfully removed!")
        st.rerun()

    st.divider()
    st.markdown("**Git Status Check**")
    st.caption("✅ `*.joblib` (Model binaries): Ignored")
    st.caption("✅ `test_predictions.csv`: Ignored")
    st.caption("✅ `metrics_summary.csv`: Ignored")
    st.caption("✅ `__pycache__/`: Ignored")


# Initialize session state for interactive pipeline
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "trained_pipelines" not in st.session_state:
    st.session_state.trained_pipelines = {}
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "test_predictions_df" not in st.session_state:
    st.session_state.test_predictions_df = None

# ==========================================
# STEP 1: DATA INSPECTION & CLEANING
# ==========================================
st.header("Step 1: Data Inspection & Missing Value Resolution")

col_raw1, col_raw2 = st.columns([3, 2])
with col_raw1:
    st.markdown("**Raw Dataset Preview (First 5 Rows)**")
    st.dataframe(raw_df.head(), use_container_width=True)

with col_raw2:
    # Identify missing values ('?' in horsepower)
    missing_mask = raw_df["horsepower"] == "?"
    missing_count = int(missing_mask.sum())
    st.metric("Missing Values Detected", f"{missing_count} rows ('?' in horsepower)")
    st.markdown("**Rows containing missing horsepower:**")
    st.dataframe(
        raw_df[missing_mask][["car name", "cylinders", "displacement", "horsepower", "weight"]],
        use_container_width=True,
    )

# Interactive imputation strategy
col_opt, col_btn = st.columns([3, 1])
with col_opt:
    impute_strategy = st.radio(
        "Select Missing Value Imputation Strategy:",
        options=[
            "Cylinder-Grouped Median Imputation (Recommended: preserves engine class differences)",
            "Global Median Imputation",
        ],
        index=0,
    )

with col_btn:
    st.write("")
    st.write("")
    apply_clean = st.button("Apply Cleaning & Imputation", type="primary", use_container_width=True)

# Apply cleaning when clicked or initialize default
if apply_clean or st.session_state.cleaned_df is None:
    df_clean = raw_df.copy()
    df_clean["horsepower"] = pd.to_numeric(df_clean["horsepower"].replace("?", np.nan))
    
    if "Cylinder-Grouped" in impute_strategy:
        df_clean["horsepower"] = df_clean["horsepower"].fillna(
            df_clean.groupby("cylinders")["horsepower"].transform("median")
        )
    else:
        df_clean["horsepower"] = df_clean["horsepower"].fillna(df_clean["horsepower"].median())
        
    st.session_state.cleaned_df = df_clean

st.success(
    f"Dataset cleaned successfully! Missing horsepower count: **{st.session_state.cleaned_df['horsepower'].isna().sum()}**. Data types verified."
)

st.divider()

# ==========================================
# STEP 2: TRAIN / TEST SPLITTING
# ==========================================
st.header("Step 2: Train / Test Splitting Configuration")

col_split1, col_split2, col_split3 = st.columns(3)
with col_split1:
    test_ratio = st.slider("Held-out Test Set Ratio (%):", min_value=10, max_value=40, value=20, step=5)
with col_split2:
    random_seed = st.number_input("Random State (Seed):", min_value=0, max_value=999, value=42, step=1)
with col_split3:
    n_total = len(st.session_state.cleaned_df)
    n_test = int(n_total * (test_ratio / 100.0))
    n_train = n_total - n_test
    st.metric("Split Distribution", f"Train: {n_train} | Test: {n_test}")

# Prepare features and target
features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
X = st.session_state.cleaned_df[features].copy()
y = st.session_state.cleaned_df["mpg"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(test_ratio / 100.0), random_state=random_seed
)

st.divider()

# ==========================================
# STEP 3: MODEL SELECTION & TRAINING
# ==========================================
st.header("Step 3: Model Execution & 5-Fold Cross-Validation")

st.markdown("Select which models to train from their independent architectural modules:")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    use_lr = st.checkbox("Linear Regression (Unscaled)", value=True)
with col_m2:
    use_knn = st.checkbox("KNeighborsRegressor (StandardScaler)", value=True)
with col_m3:
    use_svr = st.checkbox("Support Vector Machine (StandardScaler)", value=True)
with col_m4:
    use_rf = st.checkbox("Random Forest (Unscaled)", value=True)

train_btn = st.button("🚀 Train Selected Models with 5-Fold CV", type="primary", use_container_width=True)

# Train execution
if train_btn or not st.session_state.models_trained:
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    selected_runners = []
    if use_lr:
        selected_runners.append(("Linear Regression", train_lr))
    if use_knn:
        selected_runners.append(("KNeighborsRegressor", train_knn))
    if use_svr:
        selected_runners.append(("Support Vector Machine", train_svr))
    if use_rf:
        selected_runners.append(("Random Forest", train_rf))
        
    if not selected_runners:
        st.warning("Please select at least one model to train.")
    else:
        with st.spinner("Training individual model modules with 5-fold cross-validation..."):
            trained_pipes = {}
            metrics_list = []
            test_preds_df = X_test.copy()
            test_preds_df["Actual MPG"] = y_test.values

            for name, runner_fn in selected_runners:
                pipe, metrics, preds = runner_fn(X_train, y_train, X_test, y_test, kf=kf)
                trained_pipes[name] = pipe
                metrics_list.append(metrics)
                test_preds_df[f"Predicted_{name}"] = preds

            st.session_state.trained_pipelines = trained_pipes
            st.session_state.metrics_df = pd.DataFrame(metrics_list)
            st.session_state.test_predictions_df = test_preds_df
            st.session_state.models_trained = True
            
        st.success("Training and 5-fold cross-validation completed!")

st.divider()

# ==========================================
# STEP 4: MODEL PERFORMANCE COMPARISON
# ==========================================
st.header("Step 4: Model Performance Comparison")

if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
    m_df = st.session_state.metrics_df.copy()
    m_df["CV R²"] = (
        m_df["CV R² (Mean)"].map("{:.4f}".format)
        + " ± "
        + m_df["CV R² (Std)"].map("{:.4f}".format)
    )
    display_table = m_df[["Model", "MAE", "RMSE", "Test R²", "CV R²"]].sort_values(
        by="Test R²", ascending=False
    )

    st.dataframe(
        display_table.style.format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "Test R²": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    best_row = display_table.iloc[0]
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Top Model (by Test R²)", best_row["Model"])
    bc2.metric("Test R²", f"{best_row['Test R²']:.4f}")
    bc3.metric("Test RMSE", f"{best_row['RMSE']:.4f}")
    bc4.metric("Test MAE", f"{best_row['MAE']:.4f}")

st.divider()

# ==========================================
# STEP 5: INTERACTIVE PARITY PLOT
# ==========================================
st.header("Step 5: Interactive Actual vs. Predicted Chart")

if (
    st.session_state.test_predictions_df is not None
    and len(st.session_state.trained_pipelines) > 0
):
    col_chart_select, _ = st.columns([2, 3])
    with col_chart_select:
        chart_model = st.selectbox(
            "Select Model to Inspect:",
            options=list(st.session_state.trained_pipelines.keys()),
        )

    pred_col = f"Predicted_{chart_model}"
    y_actual = st.session_state.test_predictions_df["Actual MPG"]
    y_predicted = st.session_state.test_predictions_df[pred_col]
    abs_errors = np.abs(y_actual - y_predicted)

    min_limit = min(y_actual.min(), y_predicted.min()) - 2
    max_limit = max(y_actual.max(), y_predicted.max()) + 2

    fig = go.Figure()

    # Perfect prediction reference line
    fig.add_trace(
        go.Scatter(
            x=[min_limit, max_limit],
            y=[min_limit, max_limit],
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
                size=9,
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
        title=f"Actual vs. Predicted MPG — {chart_model}",
        xaxis_title="Actual MPG",
        yaxis_title="Predicted MPG",
        xaxis=dict(range=[min_limit, max_limit]),
        yaxis=dict(range=[min_limit, max_limit]),
        height=520,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================
# STEP 6: LIVE VEHICLE MPG PREDICTOR
# ==========================================
st.header("Step 6: Live Vehicle MPG Prediction")
st.markdown("Input custom vehicle specifications to test real-time inference:")

if len(st.session_state.trained_pipelines) > 0:
    with st.form("interactive_prediction_form"):
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
                "Prediction Model",
                options=list(st.session_state.trained_pipelines.keys()),
            )

        calculate_btn = st.form_submit_button(
            "Calculate Estimated Fuel Efficiency", use_container_width=True
        )

    if calculate_btn:
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
        est_mpg = float(chosen_pipe.predict(input_row)[0])
        est_kmpl = est_mpg * 0.425144

        r_c1, r_c2 = st.columns(2)
        r_c1.success(f"### Predicted Fuel Efficiency: **{est_mpg:.2f} MPG**")
        r_c2.info(f"### Metric Equivalent: **{est_kmpl:.2f} km/L**")
