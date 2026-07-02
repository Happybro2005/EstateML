import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import config
from utils import logger

def load_raw_data(clients_path=config.RAW_CLIENTS_PATH, properties_path=config.RAW_PROPERTIES_PATH):
    """
    Loads raw clients and properties datasets, and merges them using the client reference.
    """
    logger.info("Loading raw datasets...")
    if not os.path.exists(clients_path):
        raise FileNotFoundError(f"Clients file not found: {clients_path}")
    if not os.path.exists(properties_path):
        raise FileNotFoundError(f"Properties file not found: {properties_path}")
        
    df_clients = pd.read_csv(clients_path)
    df_properties = pd.read_csv(properties_path)
    
    logger.info(f"Loaded client shape: {df_clients.shape}, property shape: {df_properties.shape}")
    
    # Merge client and property datasets on clients client_id and properties client_ref.
    # INNER join ensures we only keep sold properties linked to clients
    df_merged = pd.merge(df_clients, df_properties, left_on=config.COL_CLIENT_ID, right_on="client_ref", how="inner")
    logger.info(f"Merged dataset shape: {df_merged.shape}")
    return df_merged

def clean_data(df_merged):
    """
    Performs data cleaning:
    - Removes duplicate records.
    - Handles missing values (imputation).
    - Normalizes categorical labels.
    - Converts date_of_birth to Age.
    - Performs validation checks.
    """
    logger.info("Starting data cleaning...")
    
    # Copy to avoid modifying the original dataframe
    df = df_merged.copy()
    
    # 1. Remove duplicate records
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows. New shape: {df.shape}")
    
    # 2. Convert date_of_birth to Age
    # DOB format: YYYY-MM-DD
    if config.COL_DOB in df.columns:
        logger.info("Converting date_of_birth to Age...")
        # Parse DOB, setting invalid dates to NaT
        df[config.COL_DOB] = pd.to_datetime(df[config.COL_DOB], errors="coerce")
        
        # Base year 2026
        base_year = 2026
        df[config.COL_AGE] = base_year - df[config.COL_DOB].dt.year
        
        # Impute missing Ages with the median Age
        median_age = df[config.COL_AGE].median()
        if pd.isna(median_age):
            median_age = 40.0 # Fallback
        df[config.COL_AGE] = df[config.COL_AGE].fillna(median_age)
        
        # Age validation: Clip ages between 18 and 100
        df[config.COL_AGE] = np.clip(df[config.COL_AGE], 18, 100)
        
        # Drop date_of_birth
        df = df.drop(columns=[config.COL_DOB])
    else:
        logger.warning("date_of_birth column not found! Creating default age.")
        df[config.COL_AGE] = 40.0
        
    # 3. Handle missing values for other columns
    # For numerical columns
    num_cols_to_impute = [config.COL_PURCHASE_PRICE, config.COL_SIZE, config.COL_SATISFACTION]
    for col in num_cols_to_impute:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # For categorical columns
    cat_cols_to_impute = [config.COL_CLIENT_TYPE, config.COL_GENDER, config.COL_COUNTRY, 
                          config.COL_REGION, config.COL_REFERRAL_CHANNEL, config.COL_ACQUISITION_PURPOSE, config.COL_PROPERTY_TYPE]
    for col in cat_cols_to_impute:
        if col in df.columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            
    # 4. Normalize categorical labels (strip whitespace, ensure title case where appropriate)
    for col in cat_cols_to_impute:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Map loan_applied to binary (1 for Yes, 0 for No)
    if config.COL_LOAN_APPLIED in df.columns:
        df[config.COL_LOAN_APPLIED] = df[config.COL_LOAN_APPLIED].fillna("No").astype(str).str.strip().str.lower()
        df[config.COL_LOAN_APPLIED] = df[config.COL_LOAN_APPLIED].map({"yes": 1, "no": 0}).fillna(0).astype(int)
        
    # 5. Data validation checks
    # Price must be positive
    df[config.COL_PURCHASE_PRICE] = df[config.COL_PURCHASE_PRICE].apply(lambda x: max(10000, x))
    # Size must be positive
    df[config.COL_SIZE] = df[config.COL_SIZE].apply(lambda x: max(100, x))
    # Satisfaction score must be between 1 and 5
    df[config.COL_SATISFACTION] = np.clip(df[config.COL_SATISFACTION], 1.0, 5.0)
    
    # 6. Drop any duplicates arising after column removal or standardization
    final_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Post-processing duplicate check removed {final_rows - len(df)} duplicate rows. New shape: {df.shape}")
    
    logger.info(f"Data cleaning and validation complete. Final cleaned shape: {df.shape}")
    return df

def fit_transform_features(df_cleaned, is_training=True, scaler=None, encoder=None):
    """
    Fits and/or applies OneHotEncoder for nominal categories and StandardScaler for numeric features.
    
    Parameters:
    - df_cleaned: Cleaned pandas DataFrame.
    - is_training: If True, fits new scaler and encoder and returns them.
    - scaler: Existing StandardScaler instance (if is_training=False).
    - encoder: Existing OneHotEncoder instance (if is_training=False).
    
    Returns:
    - X_processed: Numpy array of processed features for modeling.
    - processed_feature_names: List of column names corresponding to X_processed.
    - scaler: The StandardScaler object.
    - encoder: The OneHotEncoder object.
    """
    logger.info("Transforming features (encoding and scaling)...")
    
    # Select feature columns (excluding identifiers client_id and property_id)
    id_cols = [config.COL_CLIENT_ID, config.COL_PROPERTY_ID]
    feature_cols = [col for col in df_cleaned.columns if col not in id_cols]
    
    # Separate numeric and categorical
    numeric_cols = config.NUMERICAL_COLS
    nominal_cols = config.NOMINAL_COLS
    binary_cols = [config.COL_LOAN_APPLIED]
    
    # 1. Scale numerical columns
    if is_training:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_cleaned[numeric_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for transformation of testing data.")
        X_num = scaler.transform(df_cleaned[numeric_cols])
        
    # 2. One-hot encode nominal columns
    if is_training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(df_cleaned[nominal_cols])
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided for transformation of testing data.")
        X_cat = encoder.transform(df_cleaned[nominal_cols])
        
    # Categorical feature names
    cat_feature_names = encoder.get_feature_names_out(nominal_cols).tolist()
    
    # 3. Binary columns (already numeric: 0/1)
    X_bin = df_cleaned[binary_cols].values
    
    # Combine processed arrays
    X_processed = np.hstack((X_num, X_cat, X_bin))
    processed_feature_names = numeric_cols + cat_feature_names + binary_cols
    
    logger.info(f"Feature transformation complete. Processed feature matrix shape: {X_processed.shape}")
    
    return X_processed, processed_feature_names, scaler, encoder

def preprocess_pipeline():
    """
    Executes the entire preprocessing pipeline on the raw CSV files.
    """
    # Load
    df_merged = load_raw_data()
    # Clean
    df_cleaned = clean_data(df_merged)
    # Fit/Transform
    X_processed, feature_names, scaler, encoder = fit_transform_features(df_cleaned, is_training=True)
    
    return df_cleaned, X_processed, feature_names, scaler, encoder

if __name__ == "__main__":
    df_cleaned, X_processed, feature_names, scaler, encoder = preprocess_pipeline()
    print("Cleaned shape:", df_cleaned.shape)
    print("Processed matrix shape:", X_processed.shape)
