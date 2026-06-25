import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
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

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _optimize_rf(X_train, y_train, X_val, y_val, n_trials=60):
    """Run Optuna HPO for Random Forest."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
            'max_depth': trial.suggest_int('max_depth', 8, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42,
            'n_jobs': -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return r2_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='rf_hpo')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"RF Best R² (val): {study.best_value:.6f}")
    logger.info(f"RF Best Params: {study.best_params}")
    return study.best_params


def _optimize_xgb(X_train, y_train, X_val, y_val, n_trials=80):
    """Run Optuna HPO for XGBoost with early stopping."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
        }
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        return r2_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='xgb_hpo')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"XGB Best R² (val): {study.best_value:.6f}")
    logger.info(f"XGB Best Params: {study.best_params}")
    return study.best_params


def _optimize_lgbm(X_train, y_train, X_val, y_val, n_trials=80):
    """Run Optuna HPO for LightGBM with early stopping."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        preds = model.predict(X_val)
        return r2_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='lgbm_hpo')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"LGBM Best R² (val): {study.best_value:.6f}")
    logger.info(f"LGBM Best Params: {study.best_params}")
    return study.best_params


def _optimize_ensemble_weights(rf_preds, xgb_preds, lgbm_preds, y_val):
    """Find optimal ensemble weights via grid search on validation set."""
    best_r2 = -np.inf
    best_weights = (0.0, 0.45, 0.55)  # default: RF=0, XGB=0.45, LGBM=0.55
    
    # Fine-grained grid search over weight combinations
    for w_rf in np.arange(0.0, 0.41, 0.05):
        for w_xgb in np.arange(0.0, 1.01 - w_rf, 0.05):
            w_lgbm = 1.0 - w_rf - w_xgb
            if w_lgbm < 0:
                continue
            
            ens_preds = w_rf * rf_preds + w_xgb * xgb_preds + w_lgbm * lgbm_preds
            r2 = r2_score(y_val, ens_preds)
            
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (round(w_rf, 2), round(w_xgb, 2), round(w_lgbm, 2))
    
    logger.info(f"Optimal ensemble weights — RF: {best_weights[0]}, XGB: {best_weights[1]}, LGBM: {best_weights[2]}")
    logger.info(f"Ensemble R² (val): {best_r2:.6f}")
    return best_weights, best_r2


def train_and_tune(data_path="data/raw/smart_city_traffic_data.csv", fast_mode=True):
    logger.info(f"Starting model training pipeline in {'FAST' if fast_mode else 'FULL OPTUNA HPO'} mode.")
    
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
    X_test = X_test_full[features]
    y_test = X_test_full[target]
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save Feature Engineer
    fe.save("models/feature_engineer.pkl")
    
    # 4. Train Models
    if fast_mode:
        # ---- FAST MODE: Fixed hyperparameters for quick iterations ----
        logger.info("Training Random Forest Regressor (fast mode)...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        logger.info("Training XGBoost Regressor (fast mode)...")
        xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1, tree_method='hist')
        xgb_model.fit(X_train, y_train)
        
        logger.info("Training LightGBM Regressor (fast mode)...")
        lgbm_model = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    random_state=42, n_jobs=-1, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        
        # Default ensemble weights for fast mode
        rf_weight, xgb_weight, lgbm_weight = 0.0, 0.45, 0.55
    else:
        # ---- FULL MODE: Optuna Bayesian HPO ----
        logger.info("=" * 60)
        logger.info("STARTING OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        
        # 4.1 Optimize Random Forest
        logger.info("--- Optimizing Random Forest ---")
        rf_best_params = _optimize_rf(X_train, y_train, X_val, y_val, n_trials=60)
        rf_model = RandomForestRegressor(**rf_best_params, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # 4.2 Optimize XGBoost
        logger.info("--- Optimizing XGBoost ---")
        xgb_best_params = _optimize_xgb(X_train, y_train, X_val, y_val, n_trials=80)
        xgb_model = XGBRegressor(**xgb_best_params, random_state=42, n_jobs=-1, tree_method='hist')
        xgb_model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
        
        # 4.3 Optimize LightGBM
        logger.info("--- Optimizing LightGBM ---")
        lgbm_best_params = _optimize_lgbm(X_train, y_train, X_val, y_val, n_trials=80)
        lgbm_model = LGBMRegressor(**lgbm_best_params, random_state=42, n_jobs=-1, verbose=-1)
        lgbm_model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)])
        
        # 4.4 Optimize ensemble weights
        logger.info("--- Optimizing Ensemble Weights ---")
        rf_val_preds = rf_model.predict(X_val)
        xgb_val_preds = xgb_model.predict(X_val)
        lgbm_val_preds = lgbm_model.predict(X_val)
        
        (rf_weight, xgb_weight, lgbm_weight), ens_val_r2 = _optimize_ensemble_weights(
            rf_val_preds, xgb_val_preds, lgbm_val_preds, y_val
        )
    
    # Save initial models
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    
    # 5. Evaluate on Validation Set for Comparison
    rf_val_r2 = rf_model.score(X_val, y_val)
    xgb_val_r2 = xgb_model.score(X_val, y_val)
    lgbm_val_r2 = lgbm_model.score(X_val, y_val)
    
    ens_val_preds = rf_weight * rf_model.predict(X_val) + xgb_weight * xgb_model.predict(X_val) + lgbm_weight * lgbm_model.predict(X_val)
    ens_val_r2 = r2_score(y_val, ens_val_preds)
    
    logger.info(f"Validation R² — RF: {rf_val_r2:.4f}, XGB: {xgb_val_r2:.4f}, LGBM: {lgbm_val_r2:.4f}, Ensemble: {ens_val_r2:.4f}")
    
    # 6. Retrain on combined Train+Val for maximum performance
    logger.info("Combining Train and Validation sets for final retraining...")
    combined_train_val_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_train_val = combined_train_val_df[target]
    
    # Refit FeatureEngineer on combined set to prevent target encoding leakage on test set
    fe.fit(combined_train_val_df)
    fe.save("models/feature_engineer.pkl")
    
    # Re-transform combined set and test features
    X_train_val_full = fe.transform(combined_train_val_df)
    X_train_val = X_train_val_full[features]
    
    X_test_full_retransformed = fe.transform(test_df)
    X_test = X_test_full_retransformed[features]
    y_test = X_test_full_retransformed[target]
    
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
    
    # 7. Evaluate on held-out Test Set
    rf_test_preds = rf_model.predict(X_test)
    xgb_test_preds = xgb_model.predict(X_test)
    lgbm_test_preds = lgbm_model.predict(X_test)
    ens_test_preds = rf_weight * rf_test_preds + xgb_weight * xgb_test_preds + lgbm_weight * lgbm_test_preds
    
    rf_test_r2 = r2_score(y_test, rf_test_preds)
    xgb_test_r2 = r2_score(y_test, xgb_test_preds)
    lgbm_test_r2 = r2_score(y_test, lgbm_test_preds)
    ens_test_r2 = r2_score(y_test, ens_test_preds)
    
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_preds))
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_preds))
    lgbm_test_rmse = np.sqrt(mean_squared_error(y_test, lgbm_test_preds))
    ens_test_rmse = np.sqrt(mean_squared_error(y_test, ens_test_preds))
    
    logger.info("=" * 60)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"RF   — R²: {rf_test_r2:.6f}, RMSE: {rf_test_rmse:.2f}")
    logger.info(f"XGB  — R²: {xgb_test_r2:.6f}, RMSE: {xgb_test_rmse:.2f}")
    logger.info(f"LGBM — R²: {lgbm_test_r2:.6f}, RMSE: {lgbm_test_rmse:.2f}")
    logger.info(f"ENS  — R²: {ens_test_r2:.6f}, RMSE: {ens_test_rmse:.2f}")
    logger.info(f"Ensemble Weights — RF: {rf_weight}, XGB: {xgb_weight}, LGBM: {lgbm_weight}")
    logger.info("=" * 60)
    
    # 8. Save ensemble config
    import yaml
    ensemble_config = {
        'rf_weight': float(rf_weight),
        'xgb_weight': float(xgb_weight),
        'lgbm_weight': float(lgbm_weight),
    }
    os.makedirs("config", exist_ok=True)
    with open("config/ensemble_config.yaml", "w") as f:
        yaml.dump(ensemble_config, f, default_flow_style=False)
    logger.info(f"Saved ensemble config: {ensemble_config}")
    
    # 9. Save Model Registry Metadata
    registry = {
        "last_trained": datetime.datetime.now().isoformat(),
        "fast_mode": fast_mode,
        "ensemble_weights": {
            "rf": float(rf_weight),
            "xgb": float(xgb_weight),
            "lgbm": float(lgbm_weight),
        },
        "test_metrics": {
            "rf": {"r2": rf_test_r2, "rmse": rf_test_rmse},
            "xgb": {"r2": xgb_test_r2, "rmse": xgb_test_rmse},
            "lgbm": {"r2": lgbm_test_r2, "rmse": lgbm_test_rmse},
            "ensemble": {"r2": ens_test_r2, "rmse": ens_test_rmse},
        },
        "val_metrics": {
            "rf": {"r2": rf_val_r2},
            "xgb": {"r2": xgb_val_r2},
            "lgbm": {"r2": lgbm_val_r2},
            "ensemble": {"r2": ens_val_r2},
        },
        "models": {
            "random_forest": {
                "params": {k: v for k, v in rf_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            },
            "xgboost": {
                "params": {k: v for k, v in xgb_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            },
            "lightgbm": {
                "params": {k: v for k, v in lgbm_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            }
        }
    }
    
    with open("models/model_registry.json", "w") as f:
        json.dump(registry, f, indent=4)
    logger.info("Saved model registry metadata to models/model_registry.json")
    
    # Save best params
    best_params = {
        "random_forest": {k: v for k, v in rf_model.get_params().items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
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
