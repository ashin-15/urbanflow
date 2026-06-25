import os
import joblib
import logging
import pandas as pd
import numpy as np
from src.features import FeatureEngineer, get_model_features_list
from src.ensemble import WeightedEnsemble

logger = logging.getLogger(__name__)

class TrafficPredictor:
    def __init__(self, models_dir="models", config_path="config/ensemble_config.yaml"):
        self.models_dir = models_dir
        self.config_path = config_path
        
        # Load feature engineer
        fe_path = os.path.join(models_dir, "feature_engineer.pkl")
        if not os.path.exists(fe_path):
            raise FileNotFoundError(f"FeatureEngineer not found at {fe_path}. Please train models first.")
        self.fe = FeatureEngineer.load(fe_path)
        
        # Load ensemble
        self.ensemble = WeightedEnsemble(config_path=config_path, models_dir=models_dir)
        
    def _categorize_congestion(self, demand):
        """Categorizes traffic demand into Low, Medium, High congestion levels."""
        if demand < 1500:
            return "LOW", "🟢 Green", 0.0 + (demand / 1500.0) * 33.3
        elif 1500 <= demand <= 3000:
            return "MEDIUM", "🟡 Yellow", 33.3 + ((demand - 1500) / 1500.0) * 33.3
        else:
            return "HIGH", "🔴 Red", 66.6 + min(100.0, ((demand - 3000) / 5000.0)) * 33.3

    def predict_single(self, input_dict):
        """Predicts traffic demand, congestion level, and returns lookahead warnings for a single input."""
        df_single = pd.DataFrame([input_dict])
        
        # 1. Input Validation
        required_keys = [
            "timestamp", "location", "geohash_location", "road_type", 
            "number_of_lanes", "traffic_signals", "large_vehicles_count", 
            "temperature", "humidity", "rainfall", "wind_speed", "event_indicator", "nearby_landmarks",
            "weather_condition"
        ]
        
        missing_keys = [key for key in required_keys if key not in input_dict]
        if missing_keys:
            raise ValueError(f"Input dictionary missing required keys: {missing_keys}")
            
        # 2. Run Inference for current time
        df_trans = self.fe.transform(df_single)
        features = get_model_features_list()
        X = df_trans[features]
        
        pred_demand = self.ensemble.predict(X)[0]
        # Clip negative demand to 0
        pred_demand = max(0.0, pred_demand)
        
        congestion_level, badge, score_pct = self._categorize_congestion(pred_demand)
        
        # 3. 2-Hour Lookahead Forecast for Peak Alert
        # Reconstruct inputs for T+1 and T+2 hours
        timestamp = pd.to_datetime(input_dict["timestamp"])
        
        lookahead_preds = []
        for offset in [1, 2]:
            future_time = timestamp + pd.Timedelta(hours=offset)
            future_input = input_dict.copy()
            future_input["timestamp"] = future_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Predict for future hour
            df_future = pd.DataFrame([future_input])
            df_future_trans = self.fe.transform(df_future)
            X_future = df_future_trans[features]
            future_pred = max(0.0, self.ensemble.predict(X_future)[0])
            lookahead_preds.append((future_time.hour, future_pred))
            
        # Determine Peak Alert
        # Yes if current hour is High congestion, or if any of the next 2 hours is High congestion
        peak_alert = "No"
        alert_msg = "Traffic is expected to remain normal in the next 2 hours."
        
        all_forecasts = [(timestamp.hour, pred_demand)] + lookahead_preds
        high_hours = [hr for hr, pred in all_forecasts if pred > 3000]
        
        if high_hours:
            peak_alert = "Yes"
            if len(high_hours) == 1:
                alert_msg = f"⚠️ Alert: Heavy congestion (>3000 veh/hr) expected at hour {high_hours[0]}:00."
            else:
                alert_msg = f"⚠️ Alert: Heavy congestion expected over hours {', '.join(map(str, high_hours))}:00."
        
        return {
            "predicted_demand": int(round(pred_demand)),
            "congestion_level": congestion_level,
            "badge": badge,
            "gauge_score_pct": int(round(score_pct)),
            "peak_traffic_alert": peak_alert,
            "alert_message": alert_msg,
            "forecast_2h": [
                {"hour": hr, "prediction": int(round(pred))} for hr, pred in lookahead_preds
            ]
        }

def predict(input_dict, models_dir="models", config_path="config/ensemble_config.yaml"):
    """Wrapper function to instantiate predictor and run inference."""
    predictor = TrafficPredictor(models_dir=models_dir, config_path=config_path)
    return predictor.predict_single(input_dict)
