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

    def _calculate_accident_risk(self, input_dict, pred_demand, congestion_level):
        """Calculates a heuristic accident risk score (0-100%) and risk level."""
        base_risk = 5.0
        
        # Weather impact
        weather = input_dict.get("weather_condition", "Clear")
        if weather in ["Rainy", "Snowy"]:
            base_risk += 15.0
        elif weather in ["Foggy", "Thunderstorm", "Stormy"]:
            base_risk += 25.0
            
        # Congestion impact
        if congestion_level == "MEDIUM":
            base_risk += 10.0
        elif congestion_level == "HIGH":
            base_risk += 25.0
            
        # Event/Landmark impact
        if input_dict.get("event_indicator", 0) == 1:
            base_risk += 10.0
            
        # Vehicle ratio impact
        lanes = max(1, input_dict.get("number_of_lanes", 1))
        large_vehicles = input_dict.get("large_vehicles_count", 0)
        ratio = large_vehicles / lanes
        if ratio > 15:
            base_risk += 10.0
            
        risk_score = min(98.0, base_risk + np.random.uniform(-2, 2))  # add slight noise
        
        if risk_score < 30:
            level = "Low"
        elif risk_score < 60:
            level = "Moderate"
        else:
            level = "High"
            
        return int(round(risk_score)), level

    def _get_route_recommendations(self, input_dict):
        """Generates alternative route recommendations by holding time/location constant."""
        current_road = input_dict.get("road_type")
        all_road_types = ["Highway", "Arterial", "Collector", "Local Street", "Residential Street"]
        alternatives = [rt for rt in all_road_types if rt != current_road]
        
        features = get_model_features_list()
        recommendations = []
        
        for alt_road in alternatives:
            alt_input = input_dict.copy()
            alt_input["road_type"] = alt_road
            # Slightly alter lanes/signals based on road type to simulate reality
            if alt_road == "Highway":
                alt_input["number_of_lanes"] = max(3, alt_input["number_of_lanes"])
                alt_input["traffic_signals"] = 0
            elif alt_road == "Local Street" or alt_road == "Residential Street":
                alt_input["number_of_lanes"] = min(2, alt_input["number_of_lanes"])
                alt_input["traffic_signals"] = 1
                
            df_alt = pd.DataFrame([alt_input])
            df_alt_trans = self.fe.transform(df_alt)
            X_alt = df_alt_trans[features]
            alt_pred = max(0.0, self.ensemble.predict(X_alt)[0])
            
            recommendations.append({
                "road_type": alt_road,
                "predicted_demand": int(round(alt_pred))
            })
            
        # Sort by lowest demand
        recommendations = sorted(recommendations, key=lambda x: x["predicted_demand"])
        return recommendations[:2]  # Return top 2 alternatives

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
        
        # 4. Calculate Accident Risk
        risk_score, risk_level = self._calculate_accident_risk(input_dict, pred_demand, congestion_level)
        
        # 5. Route Recommendations
        alternative_routes = self._get_route_recommendations(input_dict)
        
        return {
            "predicted_demand": int(round(pred_demand)),
            "congestion_level": congestion_level,
            "badge": badge,
            "gauge_score_pct": int(round(score_pct)),
            "peak_traffic_alert": peak_alert,
            "alert_message": alert_msg,
            "forecast_2h": [
                {"hour": hr, "prediction": int(round(pred))} for hr, pred in lookahead_preds
            ],
            "accident_risk_score": risk_score,
            "accident_risk_level": risk_level,
            "alternative_routes": alternative_routes
        }

def predict(input_dict, models_dir="models", config_path="config/ensemble_config.yaml"):
    """Wrapper function to instantiate predictor and run inference."""
    predictor = TrafficPredictor(models_dir=models_dir, config_path=config_path)
    return predictor.predict_single(input_dict)
