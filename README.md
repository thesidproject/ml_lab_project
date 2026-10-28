# ML Lab Project SEM - 5 

## Predictive Analysis of Vehicle Fuel Efficiency (Auto MPG)

A reproducible machine learning project for predicting vehicle fuel efficiency (Miles Per Gallon) using the classic Auto MPG dataset.

---

## 1. Project Overview

- **Task**: Supervised Regression
- **Target Variable**: `mpg` (Fuel efficiency in Miles Per Gallon)
- **Features**: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, `origin`
- **Models Benchmarked**:
  1. **Linear Regression**
  2. **K-Nearest Neighbors Regressor (`KNeighborsRegressor`)**
  3. **Support Vector Machine (`SVR`)**
  4. **Random Forest Regressor (`RandomForestRegressor`)**

---

## 2. Directory Structure

```
project/
├── data/
│   ├── auto-mpg.csv               # Raw dataset
│   └── test_predictions.csv      # Actual vs. predicted values for held-out test set
├── models/
│   ├── linear_regression.joblib       # Serialized trained pipeline
│   ├── kneighbors_regressor.joblib    # Serialized trained pipeline
│   ├── support_vector_machine.joblib  # Serialized trained pipeline
│   ├── random_forest.joblib           # Serialized trained pipeline
│   ├── evaluation_metrics.json        # Metric results
│   └── metrics_summary.csv            # Tabular performance results
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py          # Imputation, encoding, and data splitting
│   └── train_models.py           # Selective scaling, 5-fold CV, test evaluation
├── app.py                         # Streamlit Web Application
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 3. Data Cleaning & Selective Scaling Strategy

1. **Handling Missing Values**:
   - The `horsepower` column contains 6 missing entries marked as `'?'`.
   - Missing values are imputed using the **median horsepower of the vehicle's cylinder group** rather than a simple global mean.
2. **Outlier Preservation**:
   - Outliers are retained in the dataset to preserve valid physical performance characteristics of high-displacement vehicles.
3. **Categorical Encoding**:
   - `origin` (1: USA, 2: Europe, 3: Japan) is one-hot encoded using Scikit-Learn's `OneHotEncoder(categories=[[1, 2, 3]], drop='first')`.
4. **Selective Feature Scaling (No Unnecessary Scaling)**:
   - **KNeighborsRegressor**: Scaled with `StandardScaler` (distance calculations are sensitive to large magnitude differences).
   - **Support Vector Machine (SVR)**: Scaled with `StandardScaler` (RBF kernel distance calculations require normalized feature spaces).
   - **Linear Regression**: Unscaled (preserves direct physical interpretation of coefficients).
   - **Random Forest**: Unscaled (tree-based splitting is invariant to monotonic feature transformations).

---

## 4. Model Evaluation & Benchmark Results

All models were evaluated using **5-fold cross-validation on the 80% training set**, followed by final assessment on a **20% held-out test set** (`random_state=42`).

| Model | MAE (Lower is better) | RMSE (Lower is better) | Test R² (Closer to 1.0) | 5-Fold CV R² (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVR)** | **1.6196** | **2.0187** | **0.9242** | **0.8727 ± 0.0320** |
| **Random Forest** | **1.5989** | 2.1632 | 0.9130 | 0.8510 ± 0.0278 |
| **KNeighborsRegressor** | 1.8214 | 2.2605 | 0.9050 | 0.8429 ± 0.0164 |
| **Linear Regression** | 2.2848 | 2.8866 | 0.8450 | 0.8024 ± 0.0192 |

*Note: Non-linear models (SVR and Random Forest) demonstrated the highest predictive accuracy and variance explanation ($R^2 > 0.91$).*

---

## 5. Web Application Components

The Streamlit UI (`app.py`) provides three focused, clutter-free modules:
1. **Model Performance**: Sortable evaluation table comparing MAE, RMSE, Test R², and Cross-Validation R² across all 4 models.
2. **Interactive Comparison Chart**: Plotly parity scatter plot comparing Actual MPG vs. Predicted MPG with an ideal $y=x$ dashed reference line and color-mapped absolute error points.
3. **Single Vehicle Prediction**: Parameter input fields for vehicle specifications (`cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, `origin`), outputting instant MPG and km/L predictions from any selected model pipeline.

---

## 6. How to Run

### Step 1: Activate Environment
```bash
conda activate main
```

### Step 2: (Optional) Re-train Models & Regenerate Artifacts
```bash
python project/src/train_models.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run project/app.py
```
