# Model Evaluation Report

This report presents the final evaluation metrics of the Traffic Demand Prediction System on the held-out test dataset (18,000 records, chronological split).

## 1. Summary of Performance Metrics

| Model | R² Score | RMSE (veh/hr) | MAE (veh/hr) | MAPE (%) | Pass Status (R² ≥ 0.96) |
|---|---|---|---|---|---|
| Random Forest | 0.9582 | 298.84 | 188.11 | 10.86% | ❌ Fail |
| XGBoost | 0.9590 | 296.14 | 186.75 | 10.82% | ❌ Fail |
| LightGBM | 0.9592 | 295.44 | 187.14 | 10.90% | ❌ Fail |
| **Weighted Ensemble** | **0.9593** | **295.04** | **186.50** | **10.83%** | **❌ Fail** |

> Ensemble weights: RF: 10%, XGB: 20%, LGBM: 70%

## 2. Key Findings & Discussion

1. **Ensemble Performance:** The Weighted Ensemble achieved an R² score of **0.9593** on the held-out test set.
2. **Gradient Boosting vs. Bagging:** XGBoost (R²: 0.9590) and LightGBM (R²: 0.9592) vs Random Forest (R²: 0.9582).
3. **Temporal Validity:** The dataset was split chronologically, so these metrics represent true generalization to future timepoints.

## 3. Per-Segment Analysis

### By Time Period (Hour Flag)

| Time Period | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|
| Early Morning | 4500 | 576 | 579 | -3 | 0.9131 |
| Morning Rush | 3000 | 3001 | 2996 | 5 | 0.9312 |
| Midday | 3750 | 1932 | 1934 | -3 | 0.9163 |
| Evening Rush | 3750 | 2698 | 2707 | -9 | 0.9355 |
| Night | 3000 | 1262 | 1266 | -5 | 0.9561 |

### By Road Type

| Road Type | Count | Mean Actual | Mean Predicted | Mean Error | R² |
|---|---|---|---|---|---|
| Collector | 3572 | 1810 | 1815 | -5 | 0.9494 |
| Local Street | 3684 | 1087 | 1093 | -6 | 0.9508 |
| Residential Street | 3594 | 1788 | 1786 | 2 | 0.9533 |
| Highway | 3577 | 2656 | 2664 | -8 | 0.9575 |
| Arterial | 3573 | 1775 | 1774 | 2 | 0.9533 |


## 4. Visualizations

Actual vs. Predicted scatter plot for the ensemble model:

![Actual vs. Predicted](figures/actual_vs_predicted.png)
