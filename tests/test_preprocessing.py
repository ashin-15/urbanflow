import pytest
import pandas as pd
import numpy as np
from src.preprocessing import validate_data, clean_data, handle_outliers

@pytest.fixture
def sample_config():
    return {
        "target": "traffic_demand",
        "datetime_col": "timestamp",
        "raw_numeric_features": ["number_of_lanes", "large_vehicles_count"],
        "raw_categorical_features": ["geohash_location", "weather_condition"]
    }

@pytest.fixture
def valid_dataframe():
    return pd.DataFrame({
        "timestamp": ["2024-06-04 00:00:00", "2024-06-04 01:00:00", "2024-06-04 02:00:00"],
        "number_of_lanes": [4, 2, 3],
        "large_vehicles_count": [10, 5, 8],
        "geohash_location": ["dr5reg", "ezs42", "dr5reg"],
        "weather_condition": ["Clear", "Rainy", "Clear"],
        "traffic_demand": [1000, 500, 800]
    })

def test_validate_data_success(valid_dataframe, sample_config):
    assert validate_data(valid_dataframe, sample_config) is True

def test_validate_data_missing_column(valid_dataframe, sample_config):
    invalid_df = valid_dataframe.drop(columns=["large_vehicles_count"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(invalid_df, sample_config)

def test_clean_data_deduplication(sample_config):
    duplicated_df = pd.DataFrame({
        "timestamp": ["2024-06-04 00:00:00", "2024-06-04 00:00:00", "2024-06-04 01:00:00"],
        "number_of_lanes": [4, 4, 2],
        "large_vehicles_count": [10, 10, 5],
        "geohash_location": ["dr5reg", "dr5reg", "ezs42"],
        "weather_condition": ["Clear", "Clear", "Rainy"],
        "traffic_demand": [1000, 1000, 500]
    })
    cleaned = clean_data(duplicated_df, sample_config)
    assert len(cleaned) == 2

def test_clean_data_imputation(sample_config):
    df_with_nan = pd.DataFrame({
        "timestamp": ["2024-06-04 00:00:00", "2024-06-04 01:00:00", "2024-06-04 02:00:00"],
        "number_of_lanes": [4, np.nan, 2],
        "large_vehicles_count": [10, 6, np.nan],
        "geohash_location": ["dr5reg", "ezs42", np.nan],
        "weather_condition": ["Clear", "Rainy", "Clear"],
        "traffic_demand": [1000, 500, 800]
    })
    cleaned = clean_data(df_with_nan, sample_config)
    
    # Check that NaN columns are filled
    assert not cleaned["number_of_lanes"].isnull().any()
    assert not cleaned["large_vehicles_count"].isnull().any()
    assert not cleaned["geohash_location"].isnull().any()
    
    # Check median fill for large_vehicles_count (median of [10, 6] is 8.0)
    assert cleaned["large_vehicles_count"].iloc[2] == 8.0
    # Check mode fill for geohash_location (mode is 'dr5reg' or 'ezs42', mode returns sorted)
    assert cleaned["geohash_location"].iloc[2] in ["dr5reg", "ezs42"]

def test_handle_outliers_winsorization(sample_config):
    # Setup dataframe with outliers
    demands = [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000] # 10000 is outlier
    df = pd.DataFrame({
        "timestamp": [f"2024-06-04 {i:02d}:00:00" for i in range(10)],
        "number_of_lanes": [3]*10,
        "large_vehicles_count": [5]*10,
        "geohash_location": ["dr5reg"]*10,
        "weather_condition": ["Clear"]*10,
        "traffic_demand": demands
    })
    
    winsorized = handle_outliers(df, sample_config)
    # The 99th percentile of demands is less than 10000 (quantile 0.99 is 9190.0)
    # So the value 10000 should be capped
    assert winsorized["traffic_demand"].iloc[9] < 10000
    assert winsorized["traffic_demand"].iloc[0] >= 100
