import pytest
import os
from src.predict import TrafficPredictor, predict

@pytest.fixture
def sample_input():
    return {
        "timestamp": "2026-06-25 18:00:00",
        "location": "Airport",
        "geohash_location": "dr5reg",
        "road_type": "Highway",
        "number_of_lanes": 3,
        "traffic_signals": 1,
        "large_vehicles_count": 25,
        "temperature": 28.5,
        "humidity": 65.0,
        "rainfall": 0.0,
        "wind_speed": 4.5,
        "event_indicator": 0,
        "nearby_landmarks": "Terminal 1, Cargo Area",
        "weather_condition": "Clear"
    }

def test_predict_single_inference_flow(sample_input):
    # Only run test if models directory is populated
    if not os.path.exists("models/lgbm_model.pkl"):
        pytest.skip("Models not trained, skipping integration test.")
        
    predictor = TrafficPredictor()
    result = predictor.predict_single(sample_input)
    
    assert "predicted_demand" in result
    assert "congestion_level" in result
    assert "badge" in result
    assert "gauge_score_pct" in result
    assert "peak_traffic_alert" in result
    assert "alert_message" in result
    assert "forecast_2h" in result
    
    assert isinstance(result["predicted_demand"], int)
    assert result["congestion_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(result["forecast_2h"]) == 2
    
    # Check that wrapper also works
    wrapper_res = predict(sample_input)
    assert wrapper_res["predicted_demand"] == result["predicted_demand"]

def test_predict_missing_keys(sample_input):
    predictor = TrafficPredictor()
    invalid_input = sample_input.copy()
    del invalid_input["large_vehicles_count"]
    
    with pytest.raises(ValueError, match="Input dictionary missing required keys"):
        predictor.predict_single(invalid_input)
