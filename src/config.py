"""
config.py
=========
Single source of truth for all paths, constants, and hyperparameters
used across the Car Price Prediction project.

Centralizing configuration here means:
- No magic numbers/strings scattered across modules.
- Changing a path or hyperparameter requires editing exactly one file.
- Every module (preprocessing, train, predict, evaluate) imports from here.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "quikr_car.csv"
REFERENCE_CLEANED_PATH: Path = DATA_DIR / "raw" / "Cleaned_Car_data_reference.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "cleaned_validated.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "linear_regression_model.pkl"
COMPANY_ENCODER_PATH: Path = MODELS_DIR / "company_target_encoder.pkl"
MODEL_ENCODER_PATH: Path = MODELS_DIR / "model_target_encoder.pkl"
FEATURE_METADATA_PATH: Path = MODELS_DIR / "feature_metadata.pkl"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------
# Raw column names (exactly as they appear in quikr_car.csv)
# --------------------------------------------------------------------------
COL_NAME = "name"
COL_COMPANY = "company"
COL_YEAR = "year"
COL_PRICE = "Price"
COL_KMS_DRIVEN = "kms_driven"
COL_FUEL_TYPE = "fuel_type"
COL_MODEL = "model"  # engineered feature, extracted from COL_NAME

TARGET_COLUMN = COL_PRICE
LOG_TARGET_COLUMN = "Price_log"

# --------------------------------------------------------------------------
# Cleaning rules
# --------------------------------------------------------------------------
ASK_FOR_PRICE_TOKEN = "Ask For Price"
RARE_FUEL_TYPES_TO_DROP = ["LPG"]  # too few samples (2 rows) to be learnable
PRICE_OUTLIER_HANDLING = "log_transform"  # decided in EDA discussion with user
NAME_TRUNCATE_WORDS = 3  # first N words kept from raw listing title

# Brands whose raw name format is "Company SubBrand Model ..."
# (e.g. "Maruti Suzuki Swift") -- requires skipping one extra token
# when extracting the `model` feature.
BRAND_WITH_SUBBRAND_TOKEN = {"Maruti": "Suzuki"}

# --------------------------------------------------------------------------
# Feature engineering / encoding
# --------------------------------------------------------------------------
CATEGORICAL_TARGET_ENCODE_COLS = [COL_COMPANY, COL_MODEL]
CATEGORICAL_ONEHOT_COLS = [COL_FUEL_TYPE]
NUMERICAL_COLS = [COL_YEAR, COL_KMS_DRIVEN]

TARGET_ENCODING_SMOOTHING = 10.0  # higher = trust global mean more for rare categories

# --------------------------------------------------------------------------
# Train / test split
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
CURRENT_YEAR = 2026 # used for input validation / sanity checks in frontend
MIN_VALID_YEAR = 1990
MAX_VALID_KMS = 1_000_000
