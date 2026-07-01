# UrbanFlow: Traffic Demand Prediction System

UrbanFlow is a highly accurate Machine Learning pipeline designed to predict traffic demand across a smart city using a variety of temporal, spatial, and meteorological features. The system is built around a custom feature engineering suite and a tuned 3-model weighted ensemble (LightGBM, XGBoost, and Random Forest).

## Project Structure

```
urbanflow/
│
├── data/
│   ├── raw/                 # Raw dataset (smart_city_traffic_data.csv)
│   └── processed/           # Processed and split data files (train.csv, val.csv, test.csv)
│
├── src/
│   ├── preprocessing.py     # Data cleaning, deduplication, and imputation
│   ├── features.py          # Advanced feature engineering & target encodings
│   ├── train.py             # Model training (Fast mode & Optuna HPO mode)
│   ├── ensemble.py          # Weighted ensemble logic
│   ├── predict.py           # Inference wrapper and 2-hour forecasting
│   └── evaluate.py          # Model evaluation and metrics generation
│
├── app/
│   └── main.py              # Streamlit dashboard
│
├── config/                  # Feature and ensemble configurations
├── models/                  # Saved .pkl models, encoders, and registry
├── notebooks/               # Modular Jupyter notebooks (EDA, Training, SHAP)
├── reports/                 # Output evaluation metrics and plots
├── tests/                   # Pytest suite
└── pyproject.toml           # Project metadata and dependencies
```

---

## 1. Setup & Installation

The project uses Python 3.11+ and standard data science libraries.

**Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**
The necessary packages are listed in `pyproject.toml`. You can install them using `pip`:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib streamlit plotly pyyaml pytest pytest-cov matplotlib seaborn optuna shap
```

*Note: Ensure your raw dataset (`smart_city_traffic_data.csv`) is placed in the `data/raw/` directory before proceeding.*

---

## 2. Model Training

The training pipeline handles data preprocessing, feature engineering, and model training. It has two modes: **Fast Mode** (for quick iterations) and **Full Mode** (for maximum accuracy using Optuna Bayesian Hyperparameter Optimization).

### A. Fast Mode (Default)
Trains models using fixed, pre-determined hyperparameters. Useful for quick debugging. Takes < 30 seconds.
```bash
PYTHONPATH=. python -m src.train
```

### B. Full Optimization Mode
Runs Optuna Bayesian Hyperparameter Optimization across Random Forest (60 trials), XGBoost (80 trials), and LightGBM (80 trials) using cross-validation. It then performs a grid search to find the optimal ensemble weighting. Takes ~10-15 minutes on a standard machine.
```bash
PYTHONPATH=. python -m src.train full
```

*Upon completion, models are saved to the `models/` directory alongside `model_registry.json` and `best_params.json`.*

---

## 3. Evaluation

To evaluate the trained ensemble on the held-out test dataset (15% chronological split) and generate metrics:
```bash
PYTHONPATH=. python -m src.evaluate
```
This generates:
- Terminal output with $R^2$, RMSE, MAE, and MAPE scores.
- A comprehensive markdown report: `reports/evaluation_report.md`.
- Scatter plots: `reports/figures/actual_vs_predicted.png`.

---

## 4. Running the Dashboard

The project includes a premium, interactive Streamlit dashboard for real-time inference, what-if analysis, and historical visualizations.
```bash
PYTHONPATH=. streamlit run app/main.py
```
This will launch a local server (typically at `http://localhost:8501`).

---

## 5. Testing

The codebase maintains high test coverage using `pytest`.

**Run all unit and integration tests:**
```bash
PYTHONPATH=. python -m pytest tests/ -v
```

**Run tests with coverage report:**
```bash
PYTHONPATH=. python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 6. Jupyter Notebooks

The project provides a modular, end-to-end Jupyter Notebook workflow for exploration and explainability. Note that you will need the `shap` package installed to run the explainability notebook (`uv pip install shap` or `pip install shap`).

- `01_exploratory_data_analysis.ipynb`: Data loading, quality checks, and extensive visual EDA.
- `02_feature_engineering.ipynb`: Feature engineering pipeline, interaction features, and chronological data splitting.
- `03_model_training_and_tuning.ipynb`: Training Random Forest, XGBoost, LightGBM, and tuning ensemble weights.
- `04_model_evaluation_and_explainability.ipynb`: Test set evaluation, residual analysis, and SHAP-based feature importance.

---

## Architecture Highlights
- **Feature Engineering**: Utilizes cyclical time encodings, weather interaction compounds (temp × wind), and spatio-temporal target encodings. State is carefully managed in `src/features.py` to prevent data leakage.
- **Ensembling**: Predictions are generated via a weighted average of LightGBM, XGBoost, and Random Forest. Weights are dynamically optimized over a validation set during full training.
