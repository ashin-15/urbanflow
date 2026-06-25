import pytest
import os
import yaml
import numpy as np
from src.ensemble import WeightedEnsemble

def test_ensemble_weights_loading():
    # If config file doesn't exist, it should fallback to defaults
    ens = WeightedEnsemble(config_path="nonexistent_config.yaml", models_dir="models")
    assert ens.lgbm_weight == 0.55
    assert ens.xgb_weight == 0.45
    assert ens.rf_weight == 0.0

def test_ensemble_prediction_formula(monkeypatch):
    ens = WeightedEnsemble(models_dir="models")
    
    # Mock individual models' predict calls
    class MockModel:
        def __init__(self, val):
            self.val = val
        def predict(self, X):
            return np.array([self.val] * len(X))
            
    ens.lgbm_model = MockModel(100.0)
    ens.xgb_model = MockModel(200.0)
    ens.lgbm_weight = 0.60
    ens.xgb_weight = 0.40
    ens.rf_weight = 0.0
    
    # Test 2-model formula: 0.60 * 100 + 0.40 * 200 = 60 + 80 = 140
    X_dummy = np.zeros((3, 5))
    preds = ens.predict(X_dummy)
    assert np.allclose(preds, 140.0)

def test_ensemble_3_model_prediction():
    ens = WeightedEnsemble(config_path="nonexistent_config.yaml", models_dir="models")
    
    class MockModel:
        def __init__(self, val):
            self.val = val
        def predict(self, X):
            return np.array([self.val] * len(X))
    
    ens.rf_model = MockModel(100.0)
    ens.xgb_model = MockModel(200.0)
    ens.lgbm_model = MockModel(300.0)
    ens.rf_weight = 0.10
    ens.xgb_weight = 0.40
    ens.lgbm_weight = 0.50
    
    # 0.10 * 100 + 0.40 * 200 + 0.50 * 300 = 10 + 80 + 150 = 240
    X_dummy = np.zeros((3, 5))
    preds = ens.predict(X_dummy)
    assert np.allclose(preds, 240.0)

