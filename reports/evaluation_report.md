# Model Evaluation Report

This report presents the final evaluation metrics of the Traffic Demand Prediction System on the held-out test dataset (23 records, chronological split).

## 1. Summary of Performance Metrics

| Model | R² Score | RMSE (veh/hr) | MAE (veh/hr) | MAPE (%) | Pass Status (R² ≥ 0.96) |
|---|---|---|---|---|---|
| Random Forest | -0.7821 | 1717.07 | 1387.04 | 68.78% | ❌ Fail |
| XGBoost | -1.0003 | 1819.15 | 1444.99 | 79.73% | ❌ Fail |
| LightGBM | -0.7759 | 1714.08 | 1375.42 | 70.64% | ❌ Fail |
| **Weighted Ensemble** | **-0.8506** | **1749.75** | **1388.20** | **73.58%** | **❌ Fail** |

> Ensemble weights: RF: 0%, XGB: 45%, LGBM: 55%

## 2. Key Findings & Discussion

1. **Ensemble Performance:** The Weighted Ensemble achieved an R² score of **-0.8506** on the held-out test set.
2. **Gradient Boosting vs. Bagging:** XGBoost (R²: -1.0003) and LightGBM (R²: -0.7759) vs Random Forest (R²: -0.7821).
3. **Temporal Validity:** The dataset was split chronologically, so these metrics represent true generalization to future timepoints.

## 3. Per-Segment Analysis

### By Time Period (Hour Flag)

| Time Period | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|

### By Road Type

| Road Type | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|


## 4. Visualizations

Actual vs. Predicted scatter plot for the ensemble model:

![Actual vs. Predicted](figures/actual_vs_predicted.png)
