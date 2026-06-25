import os
import logging
import yaml
import pandas as pd
import numpy as np

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/feature_config.yaml"):
    """Loads feature configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def load_data(file_path):
    """Loads dataset from a CSV file."""
    if not os.path.exists(file_path):
        error_msg = f"Data file not found at {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise

def validate_data(df, config):
    """Validates presence and types of required columns."""
    required_cols = (
        config["raw_numeric_features"] +
        config["raw_categorical_features"] +
        [config["datetime_col"]]
    )
    
    # Check if target variable is in the DataFrame (only if training)
    # If doing prediction/inference, target won't be present
    target_col = config.get("target", "traffic_demand")
    if target_col in df.columns:
        required_cols.append(target_col)
    
    # Check column presence
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns in dataset: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check row counts
    if len(df) < 1000:
        logger.warning(f"Dataset row count ({len(df)}) is lower than 1,000. Data might be insufficient.")
    
    logger.info("Data validation passed successfully.")
    return True

def clean_data(df, config):
    """Performs deduplication and imputes missing values."""
    df_clean = df.copy()
    
    # Deduplication using timestamp and geohash_location as composite key
    time_col = config["datetime_col"]
    geo_col = "geohash_location"
    
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[time_col, geo_col], keep="first")
    removed_duplicates = initial_rows - len(df_clean)
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows based on {time_col} and {geo_col}.")
    
    # Missing values imputation
    # Impute numeric features with median
    numeric_cols = config["raw_numeric_features"]
    for col in numeric_cols:
        if col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"Imputed {null_count} missing values in numeric column '{col}' with median: {median_val}")
    
    # Impute categorical features with mode
    categorical_cols = config["raw_categorical_features"]
    for col in categorical_cols:
        if col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                mode_val = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                logger.info(f"Imputed {null_count} missing values in categorical column '{col}' with mode: {mode_val}")
                
    return df_clean

def handle_outliers(df, config):
    """Applies Winsorization at 1st and 99th percentiles to the target variable to limit outlier impact."""
    df_out = df.copy()
    target_col = config.get("target", "traffic_demand")
    
    if target_col in df_out.columns:
        lower_bound = df_out[target_col].quantile(0.01)
        upper_bound = df_out[target_col].quantile(0.99)
        
        # Log outlier counts
        outliers_lower = (df_out[target_col] < lower_bound).sum()
        outliers_upper = (df_out[target_col] > upper_bound).sum()
        logger.info(f"Outlier boundaries for {target_col}: [{lower_bound}, {upper_bound}]. "
                    f"Winsorizing {outliers_lower} values below lower bound and {outliers_upper} values above upper bound.")
        
        # Winsorize
        df_out[target_col] = np.clip(df_out[target_col], lower_bound, upper_bound)
        
    return df_out

def preprocess_pipeline(file_path, config_path="config/feature_config.yaml"):
    """Runs the full preprocessing pipeline on raw data."""
    config = load_config(config_path)
    df = load_data(file_path)
    validate_data(df, config)
    df_clean = clean_data(df, config)
    df_processed = handle_outliers(df_clean, config)
    logger.info(f"Preprocessing completed successfully. Preprocessed shape: {df_processed.shape}")
    return df_processed

if __name__ == "__main__":
    import sys
    data_file = "data/raw/smart_city_traffic_data.csv"
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    try:
        df_preprocessed = preprocess_pipeline(data_file)
        # Create output directory for processed files
        os.makedirs("data/processed", exist_ok=True)
        df_preprocessed.to_csv("data/processed/preprocessed_data.csv", index=False)
        logger.info("Saved preprocessed data to data/processed/preprocessed_data.csv")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
