import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CLIENTS_PATH = os.path.join(DATA_DIR, "clients.csv")
RAW_PROPERTIES_PATH = os.path.join(DATA_DIR, "properties.csv")
PROCESSED_CLIENTS_PATH = os.path.join(DATA_DIR, "processed_clients.csv")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pkl")

# Reports paths
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORT_PATH = os.path.join(REPORTS_DIR, "report.md")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Ensure all directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Random Seed
RANDOM_STATE = 42

# Column Names - Client Dataset
COL_CLIENT_ID = "client_id"
COL_CLIENT_TYPE = "client_type"
COL_GENDER = "gender"
COL_COUNTRY = "country"
COL_REGION = "region"
COL_DOB = "date_of_birth"
COL_ACQUISITION_PURPOSE = "acquisition_purpose"
COL_LOAN_APPLIED = "loan_applied"
COL_REFERRAL_CHANNEL = "referral_channel"
COL_SATISFACTION = "satisfaction_score"
COL_AGE = "age"

# Column Names - Properties Dataset
COL_PROPERTY_ID = "listing_id"
COL_PROPERTY_TYPE = "unit_category"
COL_PURCHASE_PRICE = "sale_price"
COL_SIZE = "floor_area_sqft"

# List of numerical columns for scaling
NUMERICAL_COLS = [COL_AGE, COL_PURCHASE_PRICE, COL_SIZE, COL_SATISFACTION]

# List of nominal categorical columns for one-hot encoding
NOMINAL_COLS = [COL_CLIENT_TYPE, COL_GENDER, COL_COUNTRY, COL_REGION, COL_REFERRAL_CHANNEL, COL_ACQUISITION_PURPOSE, COL_PROPERTY_TYPE]

# Cluster mapping constants for manual profiling evaluation
CLUSTER_NAME_FIRST_TIME = "First-Time Buyers"
CLUSTER_NAME_GLOBAL = "Global Investors"
CLUSTER_NAME_CORPORATE = "Corporate Buyers"
CLUSTER_NAME_LUXURY = "Luxury Investors"
