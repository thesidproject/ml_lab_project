# Predictive Analysis of Vehicle Fuel Efficiency (Auto MPG)

A modular, reproducible machine learning system and interactive analytics studio for vehicle fuel efficiency (Miles Per Gallon) prediction using the Auto MPG dataset.

---

## 1. Project Overview

- **Task**: Supervised Regression
- **Target Variable**: `mpg` (Fuel efficiency in Miles Per Gallon)
- **Features**: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, `origin`
- **Architectural Design**: Each model is encapsulated in its own isolated module with dedicated scaling, pipeline construction, and standalone execution capability.

---

## 2. Modular Directory Structure

```
project/
├── data/
│   ├── auto-mpg.csv               # Raw dataset
│   └── test_predictions.csv      # Actual vs. predicted results on test split
├── saved_models/                  # Serialized Model Artifacts (Ignored in Git)
│   ├── linear_regression/         # Linear Regression serialized artifact
│   ├── kneighbors_regressor/      # KNN Regressor serialized artifact
│   ├── support_vector_machine/    # SVR serialized artifact
│   ├── random_forest/             # Random Forest serialized artifact
│   ├── evaluation_metrics.json    # JSON metrics summary
│   └── metrics_summary.csv        # Tabular performance results
├── src/                           # Model & Pipeline Source Code
│   ├── __init__.py
│   ├── data_cleaning.py           # Imputation, encoding, and data splitting
│   ├── linear_regression/         # Linear Regression Source Code
│   │   ├── __init__.py
│   │   └── model.py
│   ├── kneighbors_regressor/      # KNN Regressor Source Code
│   │   ├── __init__.py
│   │   └── model.py
│   ├── support_vector_machine/    # SVR Source Code
│   │   ├── __init__.py
│   │   └── model.py
│   ├── random_forest/             # Random Forest Source Code
│   │   ├── __init__.py
│   │   └── model.py
│   ├── train_all.py               # Multi-model orchestrator
│   └── train_models.py            # Entry point compatibility script
├── app.py                         # Interactive Step-by-Step Streamlit Studio
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git configuration
└── README.md                      # Documentation
```

---

## 3. Interactive Web Studio Workflow (`app.py`)

Rather than showing static pre-computed outputs, the web interface operates as a live, hands-on ML studio:

1. **Step 1: Data Inspection & Missing Value Resolution**:
   - Displays the raw dataset.
   - Highlights the 6 missing values (`'?'`) in the `horsepower` column and previews the affected vehicles.
   - Lets the user choose between **Cylinder-Grouped Median Imputation** (preserves engine class variance) and Global Imputation, with instant before/after validation.
2. **Step 2: Train / Test Splitting Configuration**:
   - Interactive slider for Held-out Test Set percentage (10% to 40%).
   - Random seed input for full reproducibility.
   - Live sample count indicator (e.g. 318 training / 80 testing).
3. **Step 3: Model Execution & 5-Fold Cross-Validation**:
   - Checkboxes to include/exclude any of the 4 models.
   - User clicks **"🚀 Train Selected Models with 5-Fold CV"** to trigger live training across the separate model modules with a visual progress spinner.
4. **Step 4: Model Performance Comparison**:
   - Sortable metrics table comparing **MAE, RMSE, Test R², and 5-Fold CV R² (Mean ± Std)**.
   - Metric callout highlighting the best-performing model on the chosen split.
5. **Step 5: Interactive Parity Chart**:
   - Plotly scatter plot comparing Actual vs. Predicted MPG with an ideal $y=x$ dashed reference line and color-mapped absolute error tooltips.
6. **Step 6: Live Vehicle MPG Prediction**:
   - Real-time parameter inputs for custom vehicle specifications, returning instantaneous predictions in both **MPG** and **km/L**.

---

## 4. Selective Scaling Strategy

| Model | Module Path | Scaling Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | `src/linear_regression/model.py` | **Unscaled** | Preserves direct physical interpretability of coefficients |
| **KNeighborsRegressor** | `src/kneighbors_regressor/model.py` | **StandardScaler** | Distance metric is sensitive to feature magnitude differences |
| **Support Vector Machine** | `src/support_vector_machine/model.py` | **StandardScaler** | RBF kernel requires standardized feature space |
| **Random Forest** | `src/random_forest/model.py` | **Unscaled** | Decision tree splits are invariant to monotonic scaling |

---

## 5. How to Run

### Step 1: Activate Environment
```bash
conda activate main
```

### Step 2: (Optional) Run Individual Models Standalone
Any model can be trained and tested individually:
```bash
python project/src/linear_regression/model.py
python project/src/kneighbors_regressor/model.py
python project/src/support_vector_machine/model.py
python project/src/random_forest/model.py
```

Or run all models simultaneously:
```bash
python project/src/train_all.py
```

### Step 3: Launch the Interactive Studio
```bash
streamlit run project/app.py
```
