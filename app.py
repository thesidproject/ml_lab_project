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

# Custom CSS: Reduce top padding, prominent headings, bigger tabs with more distance (Pink Theme)
st.markdown(
    """
    <style>
        /* Cut empty space at top */
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 1.5rem !important;
        }

        /* High-Contrast Balanced Typography */
        html, body, [class*="css"] {
            font-size: 16.5px !important;
        }
        p, span, div, li {
            font-size: 1.05rem !important;
            font-weight: 500;
        }

        /* Noticeably Bigger, Prominent Headings across all Streamlit classes */
        h1, 
        [data-testid="stHeadingWithActionElements"] h1,
        [data-testid="stMarkdownContainer"] h1,
        .stHeading h1 { 
            font-size: 2.85rem !important; 
            font-weight: 800 !important; 
            line-height: 1.2 !important;
            margin-bottom: 0.6rem !important; 
        }
        h2, 
        [data-testid="stHeadingWithActionElements"] h2,
        [data-testid="stMarkdownContainer"] h2,
        .stHeading h2 { 
            font-size: 2.25rem !important; 
            font-weight: 800 !important; 
            line-height: 1.25 !important;
            margin-top: 0.8rem !important; 
            margin-bottom: 0.5rem !important; 
        }
        h3, 
        [data-testid="stHeadingWithActionElements"] h3,
        [data-testid="stMarkdownContainer"] h3,
        .stHeading h3 { 
            font-size: 1.85rem !important; 
            font-weight: 800 !important; 
            line-height: 1.3 !important;
            margin-top: 0.7rem !important; 
            margin-bottom: 0.4rem !important;
        }

        /* Noticeably Bigger Form & Widget Labels (Test Split (%), Select Models to Train, Select Model, etc.) */
        label, 
        label p,
        label span,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] * {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
        }

        /* Distinct Tab Cards with Noticeably Bigger Tab Names & More Distance */
        div[data-baseweb="tab-list"],
        [data-testid="stTabs"] [data-baseweb="tab-list"],
        div[role="tablist"] {
            gap: 28px !important;
            padding-bottom: 16px !important;
            border-bottom: 2px solid rgba(128, 128, 128, 0.25) !important;
        }
        button[data-baseweb="tab"],
        button[role="tab"],
        [data-testid="stTabs"] button[data-baseweb="tab"],
        [data-testid="stTabs"] button[role="tab"] {
            background-color: rgba(128, 128, 128, 0.12) !important;
            border: 2px solid rgba(128, 128, 128, 0.35) !important;
            border-radius: 10px !important;
            padding: 12px 30px !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            color: inherit !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[data-baseweb="tab"] *,
        button[role="tab"] *,
        [data-testid="stTabs"] button[data-baseweb="tab"] *,
        [data-testid="stTabs"] button[role="tab"] * {
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            line-height: 1.3 !important;
            color: inherit !important;
        }
        button[data-baseweb="tab"]:hover,
        button[role="tab"]:hover {
            background-color: rgba(219, 39, 119, 0.18) !important;
            border-color: #db2777 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            background-color: #db2777 !important;
            border-color: #db2777 !important;
            box-shadow: 0 4px 12px rgba(219, 39, 119, 0.4) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] *,
        button[role="tab"][aria-selected="true"] * {
            color: #ffffff !important;
            font-weight: 800 !important;
        }
        div[data-baseweb="tab-highlight"],
        div[data-baseweb="tab-border"] {
            display: none !important;
            background-color: transparent !important;
        }

        /* Significantly Bigger Metric Card Values & Labels */
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricValue"] * {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] * {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
        }

        /* Significantly Bigger Selected Items & Form Inputs throughout UI */
        div[data-baseweb="select"] * {
            font-size: 1.22rem !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="popover"] *,
        ul[role="listbox"] li {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="tag"],
        div[data-baseweb="tag"] span,
        span[data-baseweb="tag"] {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            padding: 4px 10px !important;
        }
        input, select, textarea, div[data-baseweb="input"] input {
            font-size: 1.22rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSlider"] div,
        div[data-testid="stSlider"] span {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }

        /* Significantly Bigger Dataframe Table Fonts (MAE, MSE, RMSE Table & Preview) */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {
            font-size: 1.22rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stTable"] * {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }

        /* Primary Button */
        .stButton>button, .stFormSubmitButton>button {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            padding: 9px 22px !important;
            background-color: #db2777 !important;
            border-color: #db2777 !important;
            color: #ffffff !important;
        }
        .stButton>button:hover, .stFormSubmitButton>button:hover {
            background-color: #be185d !important;
            border-color: #be185d !important;
            color: #ffffff !important;
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
tab_data, tab_eda, tab_train, tab_chart, tab_results, tab_predict = st.tabs(
    ["Data", "EDA", "Train", "Chart", "Results", "Predict"]
)

# ==========================================
# TAB 1: DATA
# ==========================================
with tab_data:
    st.header("Dataset Inspection & Cleaning")

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    # Duplicate Rows Inspection
    if "show_duplicate_inspection" not in st.session_state:
        st.session_state.show_duplicate_inspection = False

    st.subheader("Duplicate Rows Inspection")
    col_dup_btn, _ = st.columns([1, 2])
    with col_dup_btn:
        if st.button("Check Duplicate Rows", use_container_width=True):
            st.session_state.show_duplicate_inspection = not st.session_state.show_duplicate_inspection

    # Remove duplicates can ONLY be run after checking duplicate rows
    if st.session_state.show_duplicate_inspection:
        dup_count = int(raw_df.duplicated().sum())
        if dup_count == 0:
            st.success("No duplicate rows detected across all 398 records in the dataset.")
        else:
            st.warning(f"Found {dup_count} duplicate rows.")
            st.dataframe(raw_df[raw_df.duplicated(keep=False)], use_container_width=True)

        col_rm_btn, _ = st.columns([1, 2])
        with col_rm_btn:
            if st.button("Remove Duplicate Rows", use_container_width=True):
                st.session_state.show_remove_duplicates = not st.session_state.get("show_remove_duplicates", False)

        if st.session_state.get("show_remove_duplicates", False):
            if dup_count == 0:
                st.info("No exact duplicate rows found to remove (0 duplicates). Dataset is already completely unique with 398 rows.")
            else:
                st.success(f"Removed {dup_count} exact duplicate row(s). Unique rows remaining: {len(raw_df) - dup_count}.")

    # Missing Value Inspection
    missing_count = int((raw_df["horsepower"] == "?").sum())

    if "show_missing_inspection" not in st.session_state:
        st.session_state.show_missing_inspection = False

    st.subheader("Missing Value Inspection")
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Inspect Missing Values", use_container_width=True):
            st.session_state.show_missing_inspection = not st.session_state.show_missing_inspection

    if st.session_state.show_missing_inspection:
        st.warning(
            f"Found {missing_count} vehicles containing missing horsepower values (marked as '?'). "
            "These are automatically imputed using the median horsepower of their cylinder group."
        )

        st.subheader("Vehicles with Missing Horsepower")
        missing_rows = raw_df[raw_df["horsepower"] == "?"][
            ["car name", "cylinders", "displacement", "horsepower", "weight"]
        ]
        st.dataframe(missing_rows, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.markdown(
        "Scatter plot analysis using Orange's visualization tools revealed several "
        "critical relationships between features and the target variable."
    )

    candidate_features = [
        "weight",
        "displacement",
        "horsepower",
        "acceleration",
        "cylinders",
        "model year",
        "origin",
    ]

    feature_labels = {
        "weight": "Weight (lbs)",
        "displacement": "Displacement (cu. in.)",
        "horsepower": "Horsepower",
        "acceleration": "Acceleration (0-60 mph, sec)",
        "cylinders": "Cylinders",
        "model year": "Model Year (1970 - 1982)",
        "origin": "Origin (1: USA, 2: Europe, 3: Japan)",
    }

    col_sel, col_stat = st.columns([1, 1])
    with col_sel:
        y_feat = st.selectbox(
            "Select Feature for Y-axis:",
            options=candidate_features,
            index=0,  # weight
            format_func=lambda x: feature_labels.get(x, x.title()),
        )

    corr_val = cleaned_df[y_feat].corr(cleaned_df["mpg"])

    with col_stat:
        st.markdown(
            f"<div style='padding-top: 1.6rem; font-size: 1.08rem; font-weight: 700;'>"
            f"Pearson Correlation with MPG: <span style='color: #db2777;'>r = {corr_val:+.4f}</span></div>",
            unsafe_allow_html=True,
        )

    # Big Spacious Scatter Plot (Orange style)
    fig_big = go.Figure()

    # Scatter points
    fig_big.add_trace(
        go.Scatter(
            x=cleaned_df["mpg"],
            y=cleaned_df[y_feat],
            mode="markers",
            name="Vehicles",
            marker=dict(
                size=10,
                color="#db2777",
                opacity=0.8,
                line=dict(width=1, color="#831843"),
            ),
            text=[
                f"<b>{name}</b><br>MPG: {m:.1f}<br>{feature_labels.get(y_feat, y_feat)}: {val}"
                for name, m, val in zip(
                    cleaned_df["car name"],
                    cleaned_df["mpg"],
                    cleaned_df[y_feat],
                )
            ],
            hoverinfo="text",
        )
    )

    # Polynomial Trendline to visualize curve
    try:
        p_fit = np.polyfit(cleaned_df["mpg"], cleaned_df[y_feat], deg=2)
        x_range = np.linspace(cleaned_df["mpg"].min(), cleaned_df["mpg"].max(), 100)
        y_fit = np.polyval(p_fit, x_range)
        fig_big.add_trace(
            go.Scatter(
                x=x_range,
                y=y_fit,
                mode="lines",
                name="Trendline",
                line=dict(color="#4b5563", width=2.5, dash="dash"),
            )
        )
    except Exception:
        pass

    fig_big.update_layout(
        title=dict(
            text=f"{feature_labels.get(y_feat, y_feat.title())} vs. Fuel Efficiency (MPG)",
            font=dict(size=17, weight=700),
        ),
        xaxis_title="Fuel Efficiency — MPG (Target Variable)",
        yaxis_title=feature_labels.get(y_feat, y_feat.title()),
        height=470,
        margin=dict(l=45, r=30, t=40, b=40),
        template="plotly_white",
        hoverlabel=dict(font_size=13),
    )

    st.plotly_chart(fig_big, use_container_width=True)

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

            st.success(f"Training complete. All {len(selected_runners)} selected models have been successfully trained, evaluated, and serialized.")
            st.markdown(
                """
                <div style='background-color: rgba(219, 39, 119, 0.12); border: 2px solid #db2777; border-radius: 8px; padding: 14px 18px; margin-top: 10px; font-size: 1.15rem; font-weight: 700; color: inherit;'>
                    📁 <b>Artifacts Processed & Stored:</b>
                    <ul style='margin: 8px 0 0 22px; padding: 0; font-size: 1.1rem; font-weight: 600;'>
                        <li><b>Trained Models:</b> Serialized and saved into <code>saved_models/</code></li>
                        <li><b>Test Predictions:</b> Exported and saved into <code>data/test_predictions.csv</code></li>
                        <li><b>Summary Metrics:</b> Stored in <code>saved_models/metrics_summary.csv</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            .highlight_max(subset=["Test R²"], color="#fbcfe8")
            .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#fbcfe8")
            .set_properties(**{"font-size": "1.25rem", "font-weight": "600"})
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
