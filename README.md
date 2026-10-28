# Predictive Analysis of Vehicle Fuel Efficiency (Auto MPG)

A modular, reproducible machine learning system and interactive analytics studio for vehicle fuel efficiency (Miles Per Gallon) prediction using the Auto MPG dataset.

---

## How to Run

### Step 1: Activate Environment
```bash
conda activate main
```

### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Run Individual Models Standalone
Any model can be trained and tested individually:
```bash
python src/linear_regression/model.py
python src/kneighbors_regressor/model.py
python src/support_vector_machine/model.py
python src/random_forest/model.py
```

Or run all models simultaneously:
```bash
python src/train_all.py
```

### Step 4: Launch the Interactive Studio
```bash
streamlit run app.py
```
