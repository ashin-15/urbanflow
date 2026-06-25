import os
import yaml
import joblib
import logging

logger = logging.getLogger(__name__)

class WeightedEnsemble:
    def __init__(self, config_path="config/ensemble_config.yaml", models_dir="models"):
        self.config_path = config_path
        self.models_dir = models_dir
        self.rf_model = None
        self.lgbm_model = None
        self.xgb_model = None
        self.rf_weight = 0.0
        self.lgbm_weight = 0.55
        self.xgb_weight = 0.45
        
        self.load_config()
        self.load_models()

    def load_config(self):
        """Loads weights from config file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Ensemble config not found at {self.config_path}. Using default weights: 55% LGBM, 45% XGB.")
            return
            
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            self.rf_weight = config.get("rf_weight", 0.0)
            self.lgbm_weight = config.get("lgbm_weight", 0.55)
            self.xgb_weight = config.get("xgb_weight", 0.45)
            logger.info(f"Loaded ensemble weights — RF: {self.rf_weight:.2f}, LGBM: {self.lgbm_weight:.2f}, XGB: {self.xgb_weight:.2f}")
        except Exception as e:
            logger.error(f"Error loading ensemble config: {e}")
            raise

    def load_models(self):
        """Loads RF, LGBM, and XGB models from disk."""
        rf_path = os.path.join(self.models_dir, "rf_model.pkl")
        lgbm_path = os.path.join(self.models_dir, "lgbm_model.pkl")
        xgb_path = os.path.join(self.models_dir, "xgb_model.pkl")
        
        try:
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
            else:
                logger.warning(f"Random Forest model not found at {rf_path}")
                
            if os.path.exists(lgbm_path):
                self.lgbm_model = joblib.load(lgbm_path)
            else:
                logger.warning(f"LightGBM model not found at {lgbm_path}")
                
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            else:
                logger.warning(f"XGBoost model not found at {xgb_path}")
                
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
            raise

    def predict(self, X):
        """Computes weighted average prediction of all available models."""
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("Both LightGBM and XGBoost models must be loaded to run ensemble inference.")
        
        preds = self.xgb_weight * self.xgb_model.predict(X) + self.lgbm_weight * self.lgbm_model.predict(X)
        
        # Include RF if it has weight > 0 and model is loaded
        if self.rf_weight > 0 and self.rf_model is not None:
            preds += self.rf_weight * self.rf_model.predict(X)
        
        return preds
