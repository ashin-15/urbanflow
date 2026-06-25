import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer, get_model_features_list

@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "timestamp": ["2024-06-04 00:00:00", "2024-06-04 08:00:00", "2024-06-04 18:00:00"],
        "number_of_lanes": [4, 2, 3],
        "large_vehicles_count": [10, 5, 20],
        "traffic_signals": [1, 0, 1],
        "geohash_location": ["dr5reg", "ezs42", "dr5reg"],
        "road_type": ["Highway", "Arterial", "Highway"],
        "location": ["Airport", "Downtown", "Airport"],
        "weather_condition": ["Clear", "Rainy", "Foggy"],
        "rainfall": [0.0, 10.0, 5.0],
        "humidity": [50.0, 90.0, 75.0],
        "wind_speed": [5.0, 10.0, 15.0],
        "temperature": [25.0, 30.0, 35.0],
        "event_indicator": [0, 0, 1],
        "nearby_landmarks": ["Warehouse, Port", "School, Park", "University, Hospital, Park"],
        "traffic_demand": [1000, 2000, 3000]
    })

def test_feature_engineer_fit_transform(sample_train_df):
    fe = FeatureEngineer()
    fe.fit(sample_train_df)
    
    assert fe.is_fitted is True
    assert fe.global_mean_demand_ == 2000.0
    assert fe.max_rainfall_ == 10.0
    assert fe.geohash_encoding_["dr5reg"] == 2000.0 # mean of 1000 and 3000
    
    transformed = fe.transform(sample_train_df)
    
    # Check shape/columns
    assert "hour" in transformed.columns
    assert "day_of_week" in transformed.columns
    assert "month" in transformed.columns
    assert "is_weekend" in transformed.columns
    assert "hour_flag" in transformed.columns
    assert "traffic_density_score" in transformed.columns
    assert "weather_impact_score" in transformed.columns
    assert "rush_hour_indicator" in transformed.columns
    assert "hour_sin" in transformed.columns
    assert "hour_cos" in transformed.columns
    assert "lane_signal_interaction" in transformed.columns
    assert "landmark_event_interaction" in transformed.columns
    assert "geohash_location_encoded" in transformed.columns
    assert "road_type_encoded" in transformed.columns
    
    # New advanced features
    assert "month_sin" in transformed.columns
    assert "month_cos" in transformed.columns
    assert "day_sin" in transformed.columns
    assert "day_cos" in transformed.columns
    assert "temp_wind_interaction" in transformed.columns
    assert "rainfall_humidity_interaction" in transformed.columns
    assert "vehicle_lane_ratio" in transformed.columns
    assert "rush_weekend_interaction" in transformed.columns
    assert "hour_location_encoded" in transformed.columns
    assert "dow_road_encoded" in transformed.columns
    
    # Check specific calculations
    # Landmark count
    assert transformed["nearby_landmarks_count"].iloc[0] == 2
    assert transformed["nearby_landmarks_count"].iloc[2] == 3
    
    # Hour flags
    assert transformed["hour_flag"].iloc[0] == "early_morning"
    assert transformed["hour_flag"].iloc[1] == "morning_rush"
    assert transformed["hour_flag"].iloc[2] == "evening_rush"
    
    # Rush hour indicator
    assert transformed["rush_hour_indicator"].iloc[0] == 0
    assert transformed["rush_hour_indicator"].iloc[1] == 1 # 08:00
    assert transformed["rush_hour_indicator"].iloc[2] == 1 # 18:00
    
    # Traffic density score formula: (large_vehicles / (lanes + 1)) * (1 + signals)
    # Row 0: (10 / (4 + 1)) * (1 + 1) = 2.0 * 2 = 4.0
    assert transformed["traffic_density_score"].iloc[0] == 4.0
    
    # Encodings
    assert transformed["geohash_location_encoded"].iloc[0] == 2000.0
    assert transformed["geohash_location_encoded"].iloc[1] == 2000.0

def test_feature_list_validity():
    features = get_model_features_list()
    assert len(features) > 25  # Updated for new features
    assert "geohash_location_encoded" in features
    assert "weather_impact_score" in features
    assert "month_sin" in features
    assert "hour_location_encoded" in features
    assert "dow_road_encoded" in features
    assert "vehicle_lane_ratio" in features
