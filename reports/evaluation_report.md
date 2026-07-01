# Model Evaluation Report

This report presents the final evaluation metrics of the Traffic Demand Prediction System on the held-out test dataset (18,000 records, chronological split).

## 1. Summary of Performance Metrics

| Model | R² Score | RMSE (veh/hr) | MAE (veh/hr) | MAPE (%) | Pass Status (R² ≥ 0.96) |
|---|---|---|---|---|---|
| Random Forest | 0.9572 | 302.44 | 190.27 | 10.97% | ❌ Fail |
| XGBoost | 0.9580 | 299.56 | 188.76 | 10.92% | ❌ Fail |
| LightGBM | 0.9587 | 297.22 | 188.17 | 10.95% | ❌ Fail |
| **Weighted Ensemble** | **0.9587** | **297.10** | **187.63** | **10.88%** | **❌ Fail** |

> Ensemble weights: RF: 0%, XGB: 45%, LGBM: 55%

## 2. Key Findings & Discussion

1. **Ensemble Performance:** The Weighted Ensemble achieved an R² score of **0.9587** on the held-out test set.
2. **Gradient Boosting vs. Bagging:** XGBoost (R²: 0.9580) and LightGBM (R²: 0.9587) vs Random Forest (R²: 0.9572).
3. **Temporal Validity:** The dataset was split chronologically, so these metrics represent true generalization to future timepoints.

## 3. Per-Segment Analysis

### By Time Period (Hour Flag)

| Time Period | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|
| Early Morning | 4500 | 576 | 578 | -1 | 0.9130 |
| Morning Rush | 3000 | 3001 | 3001 | -0 | 0.9294 |
| Midday | 3750 | 1932 | 1933 | -2 | 0.9157 |
| Evening Rush | 3750 | 2698 | 2706 | -8 | 0.9350 |
| Night | 3000 | 1262 | 1266 | -4 | 0.9558 |

### By Road Type

| Road Type | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|
| Collector | 3572 | 1810 | 1817 | -7 | 0.9486 |
| Local Street | 3684 | 1087 | 1090 | -3 | 0.9506 |
| Residential Street | 3594 | 1788 | 1786 | 2 | 0.9528 |
| Highway | 3577 | 2656 | 2666 | -10 | 0.9569 |
| Arterial | 3573 | 1775 | 1773 | 2 | 0.9524 |


## 4. Visualizations

Actual vs. Predicted scatter plot for the ensemble model:

![Actual vs. Predicted](figures/actual_vs_predicted.png)
