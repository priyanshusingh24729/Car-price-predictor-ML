"""
train.py
========
Trains the Linear Regression car-price model end-to-end:
  1. Load + clean raw data.
  2. Split into train/test.
  3. Fit target encoders on the TRAIN split only (leakage-safe).
  4. Build numeric feature matrices for train and test.
  5. Log-transform the target.
  6. Fit sklearn's LinearRegression.
  7. Persist the model + encoders + feature metadata to disk.

Run directly:
    python -m src.train
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from src import config
from src.preprocessing import (
    PreprocessingArtifacts,
    TargetEncoder,
    build_feature_matrix,
    load_and_clean_raw_data,
    split_data,
    validate_against_reference,
)
from src.utils import get_logger, save_pickle

logger = get_logger(__name__)


def train_model() -> dict:
    """
    Execute the full training pipeline and persist all artifacts.

    Returns:
        A dict with references to the fitted model, encoders, and the
        held-out test set (features + true/raw target), so that
        `evaluate.py` can immediately score the model without re-reading
        from disk.
    """
    # 1. Load & clean
    cleaned_df = load_and_clean_raw_data()
    is_valid = validate_against_reference(cleaned_df)
    logger.info("Cleaning matches vendor reference file: %s", is_valid)

    # 2. Split BEFORE fitting any encoder (avoids target leakage)
    train_df, test_df = split_data(cleaned_df)

    # 3. Fit target encoders on train only
    company_encoder = TargetEncoder(column=config.COL_COMPANY)
    company_encoder.fit(train_df, train_df[config.COL_PRICE])

    model_encoder = TargetEncoder(column=config.COL_MODEL)
    model_encoder.fit(train_df, train_df[config.COL_PRICE])

    fuel_type_categories = sorted(train_df[config.COL_FUEL_TYPE].unique().tolist())

    # 4. Build numeric feature matrices
    X_train = build_feature_matrix(train_df, company_encoder, model_encoder, fuel_type_categories)
    X_test = build_feature_matrix(test_df, company_encoder, model_encoder, fuel_type_categories)

    # 5. Log-transform target (decided during EDA to correct skew ~7.5)
    y_train_log = np.log1p(train_df[config.COL_PRICE])
    y_test_log = np.log1p(test_df[config.COL_PRICE])

    # 6. Fit Linear Regression
    logger.info("Training LinearRegression on %d samples, %d features", *X_train.shape)
    model = LinearRegression()
    model.fit(X_train, y_train_log)
    logger.info("Training complete. R^2 on train (log-scale): %.4f", model.score(X_train, y_train_log))

    # 7. Persist artifacts
    artifacts = PreprocessingArtifacts(
        company_encoder=company_encoder,
        model_encoder=model_encoder,
        fuel_type_categories=fuel_type_categories,
        feature_columns=list(X_train.columns),
    )
    save_pickle(model, config.MODEL_PATH)
    save_pickle(artifacts, config.FEATURE_METADATA_PATH)
    logger.info("Saved model to %s", config.MODEL_PATH)
    logger.info("Saved preprocessing artifacts to %s", config.FEATURE_METADATA_PATH)

    return {
        "model": model,
        "artifacts": artifacts,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_log": y_train_log,
        "y_test_log": y_test_log,
        "test_df_raw": test_df,
    }


if __name__ == "__main__":
    train_model()
