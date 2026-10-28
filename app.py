import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Auto MPG Fuel Efficiency Analytics",
    page_icon="🚗",
    layout="wide",
)

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_performance_data():
    """Load evaluation metrics and test set predictions."""
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.csv")
    predictions_path = os.path.join(DATA_DIR, "test_predictions.csv")

    metrics_df = pd.read_csv(metrics_path)
    predictions_df = pd.read_csv(predictions_path)
    return metrics_df, predictions_df


@st.cache_resource
def load_trained_models():
    """Load saved Scikit-Learn pipelines."""
    models = {
        "Linear Regression": joblib.load(
            os.path.join(MODELS_DIR, "linear_regression.joblib")
        ),
        "KNeighborsRegressor": joblib.load(
            os.path.join(MODELS_DIR, "kneighbors_regressor.joblib")
        ),
        "Support Vector Machine": joblib.load(
            os.path.join(MODELS_DIR, "support_vector_machine.joblib")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODELS_DIR, "random_forest.joblib")
        ),
    }
    return models


# Load data and models
metrics_df, predictions_df = load_performance_data()
models = load_trained_models()

# App Header
st.title("Predictive Analysis of Vehicle Fuel Efficiency (Auto MPG)")
st.markdown("Performance comparison and predictive inference across four regression models.")

# --- 1. Model Performance Section ---
st.subheader("1. Model Performance Comparison")
st.markdown(
    "Evaluation across 5-fold cross-validation on the training set and held-out test set metrics."
)

# Format metrics table for clean presentation
display_df = metrics_df.copy()
display_df["CV R²"] = (
    display_df["CV R² (Mean)"].map("{:.4f}".format)
    + " ± "
    + display_df["CV R² (Std)"].map("{:.4f}".format)
)
display_table = display_df[
    ["Model", "MAE", "RMSE", "Test R²", "CV R²"]
].sort_values(by="Test R²", ascending=False)

st.dataframe(
    display_table.style.format(
        {"MAE": "{:.4f}", "RMSE": "{:.4f}", "Test R²": "{:.4f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

# Best Model Callout
best_model_row = display_table.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Model (by Test R²)", best_model_row["Model"])
c2.metric("Test R²", f"{best_model_row['Test R²']:.4f}")
c3.metric("Test RMSE", f"{best_model_row['RMSE']:.4f}")
c4.metric("Test MAE", f"{best_model_row['MAE']:.4f}")

st.divider()

# --- 2. Interactive Chart: Actual vs. Predicted MPG ---
st.subheader("2. Actual vs. Predicted MPG Comparison")

col_select, _ = st.columns([2, 3])
with col_select:
    selected_model = st.selectbox(
        "Select Model to Inspect:",
        options=list(models.keys()),
        index=2,  # Default to Support Vector Machine
    )

pred_col = f"Predicted_{selected_model}"
y_true = predictions_df["Actual MPG"]
y_pred = predictions_df[pred_col]
abs_error = np.abs(y_true - y_pred)

# Build Plotly parity plot
min_val = min(y_true.min(), y_pred.min()) - 2
max_val = max(y_true.max(), y_pred.max()) + 2

fig = go.Figure()

# Perfect prediction line (y = x)
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
        x=y_true,
        y=y_pred,
        mode="markers",
        name=selected_model,
        marker=dict(
            size=9,
            color=abs_error,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Absolute Error"),
            opacity=0.85,
        ),
        text=[
            f"Actual: {a:.1f} MPG<br>Predicted: {p:.1f} MPG<br>Error: {e:.2f} MPG"
            for a, p, e in zip(y_true, y_pred, abs_error)
        ],
        hoverinfo="text",
    )
)

fig.update_layout(
    title=f"Actual vs. Predicted Fuel Efficiency — {selected_model}",
    xaxis_title="Actual MPG",
    yaxis_title="Predicted MPG",
    xaxis=dict(range=[min_val, max_val]),
    yaxis=dict(range=[min_val, max_val]),
    height=550,
    template="plotly_white",
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 3. Vehicle Prediction Section ---
st.subheader("3. Single Vehicle MPG Prediction")
st.markdown("Specify vehicle technical specifications to generate an estimated fuel efficiency.")

# Input Form
with st.form("prediction_form"):
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
    with row1_c2:
        displacement = st.number_input(
            "Displacement (cu. in.)", min_value=60.0, max_value=500.0, value=150.0, step=5.0
        )
    with row1_c3:
        horsepower = st.number_input(
            "Horsepower", min_value=40.0, max_value=300.0, value=95.0, step=1.0
        )

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        weight = st.number_input(
            "Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2800.0, step=50.0
        )
    with row2_c2:
        acceleration = st.number_input(
            "Acceleration (0-60 mph, sec)",
            min_value=5.0,
            max_value=30.0,
            value=15.5,
            step=0.5,
        )
    with row2_c3:
        model_year = st.slider(
            "Model Year (1970 - 1982)", min_value=70, max_value=82, value=76
        )

    row3_c1, row3_c2 = st.columns([1, 2])
    with row3_c1:
        origin_label = st.selectbox(
            "Origin",
            options=["USA (1)", "Europe (2)", "Japan (3)"],
            index=0,
        )
        origin_map = {"USA (1)": 1, "Europe (2)": 2, "Japan (3)": 3}
        origin_val = origin_map[origin_label]

    with row3_c2:
        pred_model_choice = st.selectbox(
            "Prediction Model",
            options=list(models.keys()),
            index=3,  # Default to Random Forest
        )

    submitted = st.form_submit_button("Calculate Estimated Fuel Efficiency", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "cylinders": cylinders,
                "displacement": displacement,
                "horsepower": horsepower,
                "weight": weight,
                "acceleration": acceleration,
                "model year": model_year,
                "origin": origin_val,
            }
        ]
    )

    chosen_pipeline = models[pred_model_choice]
    predicted_mpg = float(chosen_pipeline.predict(input_data)[0])
    predicted_kmpl = predicted_mpg * 0.425144

    res_c1, res_c2 = st.columns(2)
    res_c1.success(f"### Predicted Fuel Efficiency: **{predicted_mpg:.2f} MPG**")
    res_c2.info(f"### Equivalent: **{predicted_kmpl:.2f} km/L**")
