import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.preprocessing import preprocess_pipeline
from src.features import FeatureEngineer, get_model_features_list

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_and_tune(data_path="data/raw/smart_city_traffic_data.csv", fast_mode=True):
    logger.info(f"Starting model training pipeline in {'FAST' if fast_mode else 'FULL'} mode.")
    
    # 1. Load and Preprocess Data
    df_preprocessed = preprocess_pipeline(data_path)
    
    # Sort chronologically to prevent temporal data leakage
    df_preprocessed['timestamp'] = pd.to_datetime(df_preprocessed['timestamp'])
    df_preprocessed = df_preprocessed.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Train / Val / Test Split
    n_rows = len(df_preprocessed)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    
    train_df = df_preprocessed.iloc[:train_end].copy()
    val_df = df_preprocessed.iloc[train_end:val_end].copy()
    test_df = df_preprocessed.iloc[val_end:].copy()
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 3. Fit Feature Engineer strictly on Training Set
    fe = FeatureEngineer()
    fe.fit(train_df)
    
    # Transform all splits
    X_train_full = fe.transform(train_df)
    X_val_full = fe.transform(val_df)
    X_test_full = fe.transform(test_df)
    
    # Save processed splits for validation/evaluation
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    logger.info("Saved raw train, val, and test splits to data/processed/")
    
    # Get model features
    features = get_model_features_list()
    target = "traffic_demand"
    
    X_train = X_train_full[features]
    y_train = X_train_full[target]
    X_val = X_val_full[features]
    y_val = X_val_full[target]
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save Feature Engineer
    fe.save("models/feature_engineer.pkl")
    
    # 4. Train Models
    
    # --- 4.1 Random Forest Regressor ---
    logger.info("Training Random Forest Regressor...")
    if fast_mode:
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    else:
        # Use recommended tuned parameters
        rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
        
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "models/rf_model.pkl")
    logger.info("Random Forest Regressor trained and saved.")
    
    # --- 4.2 XGBoost Regressor ---
    logger.info("Training XGBoost Regressor...")
    if fast_mode:
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    else:
        # Use recommended tuned parameters
        xgb_model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1
        )
        
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    logger.info("XGBoost Regressor trained and saved.")
    
    # --- 4.3 LightGBM Regressor ---
    logger.info("Training LightGBM Regressor...")
    if fast_mode:
        lgbm_model = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    else:
        # Use recommended tuned parameters
        lgbm_model = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=127,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    lgbm_model.fit(X_train, y_train)
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    logger.info("LightGBM Regressor trained and saved.")
    
    # 5. Evaluate on Validation Set for Comparison
    rf_val_r2 = rf_model.score(X_val, y_val)
    xgb_val_r2 = xgb_model.score(X_val, y_val)
    lgbm_val_r2 = lgbm_model.score(X_val, y_val)
    
    logger.info(f"Validation R2 Scores (before retraining) - RF: {rf_val_r2:.4f}, XGB: {xgb_val_r2:.4f}, LGBM: {lgbm_val_r2:.4f}")
    
    # Combined retraining to maximize training size and accuracy
    logger.info("Combining Train and Validation sets for final retraining...")
    combined_train_val_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_train_val = combined_train_val_df[target]
    
    # Refit FeatureEngineer on combined set to prevent target encoding leakage on test set
    fe.fit(combined_train_val_df)
    fe.save("models/feature_engineer.pkl")
    
    # Re-transform combined set and test features
    X_train_val_full = fe.transform(combined_train_val_df)
    X_train_val = X_train_val_full[features]
    
    logger.info("Retraining final Random Forest on combined set...")
    rf_model.fit(X_train_val, y_train_val)
    joblib.dump(rf_model, "models/rf_model.pkl")
    
    logger.info("Retraining final XGBoost on combined set...")
    xgb_model.fit(X_train_val, y_train_val)
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    
    logger.info("Retraining final LightGBM on combined set...")
    lgbm_model.fit(X_train_val, y_train_val)
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    
    logger.info("Final retrained models and feature engineer saved successfully.")
    
    # 6. Save Model Registry Metadata
    registry = {
        "last_trained": datetime.datetime.now().isoformat(),
        "fast_mode": fast_mode,
        "models": {
            "random_forest": {
                "val_r2": rf_val_r2,
                "params": rf_model.get_params()
            },
            "xgboost": {
                "val_r2": xgb_val_r2,
                "params": {k: v for k, v in xgb_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            },
            "lightgbm": {
                "val_r2": lgbm_val_r2,
                "params": {k: v for k, v in lgbm_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            }
        }
    }
    
    with open("models/model_registry.json", "w") as f:
        json.dump(registry, f, indent=4)
    logger.info("Saved model registry metadata to models/model_registry.json")
    
    # Save best params
    best_params = {
        "xgboost": {k: v for k, v in xgb_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
        "lightgbm": {k: v for k, v in lgbm_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    }
    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info("Saved best parameters to models/best_params.json")
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    import sys
    fast = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "full":
        fast = False
    
    train_and_tune(fast_mode=fast)
