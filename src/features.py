import numpy as np
import pandas as pd
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.is_fitted = False
        self.geohash_encoding_ = {}
        self.road_type_encoding_ = {}
        self.location_encoding_ = {}
        self.global_mean_demand_ = 0.0
        self.max_rainfall_ = 1.0
        
        # Weather mapping
        self.weather_weights = {
            'Clear': 0,
            'Cloudy': 1,
            'Foggy': 2,
            'Rainy': 3,
            'Snowy': 4,
            'Thunderstorm': 5,
            'Stormy': 5
        }

    def _parse_landmarks_count(self, landmarks_series):
        """Helper to parse landmark strings and count them."""
        def count_items(val):
            if pd.isna(val) or not isinstance(val, str):
                return 0
            items = [item.strip() for item in val.split(',') if item.strip()]
            return len(items)
        return landmarks_series.apply(count_items)

    def fit(self, df, target_col='traffic_demand'):
        """Fits the target encodings and other training-set constants."""
        logger.info("Fitting FeatureEngineer...")
        df_fit = df.copy()
        
        # Parse datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_fit['timestamp']):
            df_fit['timestamp'] = pd.to_datetime(df_fit['timestamp'])
            
        self.global_mean_demand_ = df_fit[target_col].mean()
        
        # Fit geohash target encoding
        self.geohash_encoding_ = df_fit.groupby('geohash_location')[target_col].mean().to_dict()
        
        # Fit road_type target encoding
        self.road_type_encoding_ = df_fit.groupby('road_type')[target_col].mean().to_dict()
        
        # Fit location target encoding
        self.location_encoding_ = df_fit.groupby('location')[target_col].mean().to_dict()
        
        # Fit max rainfall for weather score scaling (avoid division by zero)
        self.max_rainfall_ = df_fit['rainfall'].max()
        if self.max_rainfall_ == 0:
            self.max_rainfall_ = 1.0
            
        self.is_fitted = True
        logger.info(f"FeatureEngineer fitted successfully. Global mean: {self.global_mean_demand_:.2f}, Max rainfall: {self.max_rainfall_:.2f}")
        return self

    def transform(self, df):
        """Applies all feature engineering and encodings to the input DataFrame."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform().")
            
        df_trans = df.copy()
        
        # Ensure timestamp is parsed
        if not pd.api.types.is_datetime64_any_dtype(df_trans['timestamp']):
            df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
            
        # 1. Core datetime components
        df_trans['hour'] = df_trans['timestamp'].dt.hour
        df_trans['day_of_week'] = df_trans['timestamp'].dt.dayofweek
        df_trans['month'] = df_trans['timestamp'].dt.month
        df_trans['is_weekend'] = df_trans['day_of_week'].isin([5, 6]).astype(int)
        
        # 2. hour_flag
        def get_hour_flag(hr):
            if 0 <= hr < 6:
                return 'early_morning'
            elif 6 <= hr < 10:
                return 'morning_rush'
            elif 10 <= hr < 15:
                return 'midday'
            elif 15 <= hr < 20:
                return 'evening_rush'
            else:
                return 'night'
        df_trans['hour_flag'] = df_trans['hour'].apply(get_hour_flag)
        
        # 3. weekend_flag (same as is_weekend)
        df_trans['weekend_flag'] = df_trans['is_weekend']
        
        # 4. rush_hour_indicator
        df_trans['rush_hour_indicator'] = df_trans['hour'].isin(list(range(7, 10)) + list(range(16, 20))).astype(int)
        
        # 5. parse landmarks count
        df_trans['nearby_landmarks_count'] = self._parse_landmarks_count(df_trans['nearby_landmarks'])
        
        # 6. traffic_density_score
        # Formula: (large_vehicles_count / (number_of_lanes + 1)) * (1 + traffic_signals)
        df_trans['traffic_density_score'] = (
            df_trans['large_vehicles_count'] / (df_trans['number_of_lanes'] + 1)
        ) * (1 + df_trans['traffic_signals'])
        
        # 7. weather_impact_score
        # Map weather condition weights, default to Clear (0) if unseen
        weather_mapped = df_trans['weather_condition'].map(self.weather_weights).fillna(0)
        df_trans['weather_impact_score'] = (
            weather_mapped * 0.4
            + (df_trans['rainfall'] / self.max_rainfall_) * 0.4
            + (df_trans['humidity'] / 100.0) * 0.2
        )
        
        # 8. Cyclical time encodings
        df_trans['hour_sin'] = np.sin(2 * np.pi * df_trans['hour'] / 24.0)
        df_trans['hour_cos'] = np.cos(2 * np.pi * df_trans['hour'] / 24.0)
        
        # 9. Interaction features
        df_trans['lane_signal_interaction'] = df_trans['number_of_lanes'] * df_trans['traffic_signals']
        df_trans['landmark_event_interaction'] = df_trans['nearby_landmarks_count'] * df_trans['event_indicator']
        
        # 10. Apply Target Encodings
        # Geohash
        df_trans['geohash_location_encoded'] = df_trans['geohash_location'].map(self.geohash_encoding_).fillna(self.global_mean_demand_)
        
        # Road Type
        df_trans['road_type_encoded'] = df_trans['road_type'].map(self.road_type_encoding_).fillna(self.global_mean_demand_)
        
        # Location
        df_trans['location_encoded'] = df_trans['location'].map(self.location_encoding_).fillna(self.global_mean_demand_)
        
        # Drop columns that are categorical or raw strings unless they are needed downstream
        # We will keep raw features in the DataFrame but the model features list will exclude them.
        
        return df_trans

    def save(self, file_path):
        """Serializes the fitted engineer to joblib file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved FeatureEngineer to {file_path}")

    @staticmethod
    def load(file_path):
        """Loads and returns a serialized FeatureEngineer."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FeatureEngineer file not found at {file_path}")
        engineer = joblib.load(file_path)
        logger.info(f"Loaded FeatureEngineer from {file_path}")
        return engineer

# Helper function to generate feature matrix for models
def get_model_features_list():
    """Returns the exact list of feature names that should be passed to the models."""
    return [
        'number_of_lanes',
        'traffic_signals',
        'large_vehicles_count',
        'temperature',
        'humidity',
        'rainfall',
        'wind_speed',
        'event_indicator',
        'nearby_landmarks_count',
        'traffic_density_score',
        'weather_impact_score',
        'rush_hour_indicator',
        'hour_sin',
        'hour_cos',
        'lane_signal_interaction',
        'landmark_event_interaction',
        'is_weekend',
        'day_of_week',
        'month',
        'hour',
        'geohash_location_encoded',
        'road_type_encoded',
        'location_encoded'
    ]
