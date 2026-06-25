import os
import yaml
import joblib
import logging

logger = logging.getLogger(__name__)

class WeightedEnsemble:
    def __init__(self, config_path="config/ensemble_config.yaml", models_dir="models"):
        self.config_path = config_path
        self.models_dir = models_dir
        self.lgbm_model = None
        self.xgb_model = None
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
            self.lgbm_weight = config.get("lgbm_weight", 0.55)
            self.xgb_weight = config.get("xgb_weight", 0.45)
            logger.info(f"Loaded ensemble weights - LGBM: {self.lgbm_weight:.2f}, XGB: {self.xgb_weight:.2f}")
        except Exception as e:
            logger.error(f"Error loading ensemble config: {e}")
            raise

    def load_models(self):
        """Loads LGBM and XGB models from disk."""
        lgbm_path = os.path.join(self.models_dir, "lgbm_model.pkl")
        xgb_path = os.path.join(self.models_dir, "xgb_model.pkl")
        
        try:
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
        """Computes weighted average prediction of LGBM and XGB models."""
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("Both LightGBM and XGBoost models must be loaded to run ensemble inference.")
            
        lgbm_preds = self.lgbm_model.predict(X)
        xgb_preds = self.xgb_model.predict(X)
        
        ensemble_preds = (self.lgbm_weight * lgbm_preds) + (self.xgb_weight * xgb_preds)
        return ensemble_preds
