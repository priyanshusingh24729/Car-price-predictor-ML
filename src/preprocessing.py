"""
preprocessing.py
=================
Reusable, leakage-safe preprocessing pipeline for the Quikr used-car dataset.

This module encodes every cleaning/encoding decision made during EDA:
  1. Parse the messy raw `quikr_car.csv` into proper dtypes
     (validated 816/816 rows to exactly match the vendor-provided
     Cleaned_Car_data.csv).
  2. Drop the 2 LPG `fuel_type` rows (too few samples to generalize).
  3. Extract a lower-cardinality `model` feature from the free-text `name`
     column (254 unique names -> ~127 unique models).
  4. Target-encode high-cardinality categoricals (`company`, `model`)
     using ONLY training-fold statistics to avoid data leakage.
  5. One-hot encode the low-cardinality `fuel_type`.
  6. Log-transform the target (`Price`) to correct heavy right-skew
     (raw skew ~7.5) instead of removing the extreme-price outlier.

Design notes:
- `TargetEncoder` is a custom scikit-learn-compatible transformer
  (implements fit/transform) so it composes cleanly with
  train/test splitting and can be pickled alongside the model.
- All cleaning functions are pure (no hidden global state), which
  makes them independently unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


# ==========================================================================
# Custom exceptions
# ==========================================================================
class DataValidationError(Exception):
    """Raised when the input dataframe doesn't meet minimum expectations."""


# ==========================================================================
# Step 1: Raw data cleaning
# ==========================================================================
class RawDataCleaner:
    """
    Cleans the raw, messy `quikr_car.csv` export into typed, valid rows.

    This class exists separately from feature engineering because
    "cleaning" (fixing broken/missing raw values) and "feature
    engineering" (deriving new modeling features) are conceptually
    different responsibilities, even though they run back-to-back.
    """

    def __init__(self) -> None:
        self.rows_dropped_: dict[str, int] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full raw-cleaning sequence.

        Args:
            df: Raw dataframe loaded directly from quikr_car.csv.

        Returns:
            A cleaned dataframe with proper dtypes and no missing values.

        Raises:
            DataValidationError: If required columns are missing.
        """
        required_cols = {
            config.COL_NAME,
            config.COL_COMPANY,
            config.COL_YEAR,
            config.COL_PRICE,
            config.COL_KMS_DRIVEN,
            config.COL_FUEL_TYPE,
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required raw columns: {missing}")

        df = df.copy()
        start_rows = len(df)
        logger.info("Starting raw cleaning with %d rows", start_rows)

        df = self._clean_year(df)
        df = self._clean_price(df)
        df = self._clean_kms_driven(df)
        df = self._drop_missing_fuel_type(df)
        df = self._truncate_name(df)
        df = self._drop_rare_fuel_types(df)

        df = df.reset_index(drop=True)
        logger.info(
            "Cleaning complete: %d -> %d rows (%d dropped)",
            start_rows,
            len(df),
            start_rows - len(df),
        )
        return df

    def _clean_year(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        mask = df[config.COL_YEAR].astype(str).str.isnumeric()
        df = df[mask].copy()
        df[config.COL_YEAR] = df[config.COL_YEAR].astype(int)
        self.rows_dropped_["invalid_year"] = before - len(df)
        return df

    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df[config.COL_PRICE] != config.ASK_FOR_PRICE_TOKEN].copy()
        df[config.COL_PRICE] = (
            df[config.COL_PRICE].astype(str).str.replace(",", "", regex=False).astype(int)
        )
        self.rows_dropped_["ask_for_price"] = before - len(df)
        return df

    def _clean_kms_driven(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        kms = df[config.COL_KMS_DRIVEN].astype(str).str.split(" ").str.get(0)
        kms = kms.str.replace(",", "", regex=False)
        valid_mask = kms.str.isnumeric()
        df = df[valid_mask].copy()
        df[config.COL_KMS_DRIVEN] = kms[valid_mask].astype(int)
        self.rows_dropped_["invalid_kms"] = before - len(df)
        return df

    def _drop_missing_fuel_type(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df[config.COL_FUEL_TYPE].isna()].copy()
        self.rows_dropped_["missing_fuel_type"] = before - len(df)
        return df

    def _truncate_name(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[config.COL_NAME] = (
            df[config.COL_NAME]
            .str.split(" ")
            .str.slice(0, config.NAME_TRUNCATE_WORDS)
            .str.join(" ")
        )
        return df

    def _drop_rare_fuel_types(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df[config.COL_FUEL_TYPE].isin(config.RARE_FUEL_TYPES_TO_DROP)].copy()
        self.rows_dropped_["rare_fuel_type"] = before - len(df)
        return df


def validate_against_reference(
    cleaned_df: pd.DataFrame, reference_path=config.REFERENCE_CLEANED_PATH
) -> bool:
    """
    Sanity-check that our cleaning logic matches the vendor-provided
    Cleaned_Car_data.csv (before the LPG-drop step, which the reference
    file does not apply).

    This is a development-time safety net, not part of the production
    inference path.

    Returns:
        True if row-for-row values match after sorting; False otherwise.
    """
    reference = pd.read_csv(reference_path).drop(columns=["Unnamed: 0"], errors="ignore")
    compare_cols = [config.COL_COMPANY, config.COL_YEAR, config.COL_PRICE, config.COL_KMS_DRIVEN]

    ours = cleaned_df.sort_values(compare_cols).reset_index(drop=True)
    theirs = reference.sort_values(compare_cols).reset_index(drop=True)

    if len(ours) != len(theirs):
        # Expected: our pipeline additionally drops LPG rows.
        logger.warning(
            "Row count differs from reference (%d vs %d) -- "
            "expected due to LPG-row removal, not an error.",
            len(ours),
            len(theirs),
        )
        return False

    return bool((ours[compare_cols].values == theirs[compare_cols].values).all())


# ==========================================================================
# Step 2: Feature engineering
# ==========================================================================
def extract_model_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a lower-cardinality `model` column from the free-text `name`
    column (e.g. "Maruti Suzuki Swift" -> "Swift", "Hyundai Santro Xing"
    -> "Santro").

    Heuristic (validated against the dataset during EDA):
    - For brands in `config.BRAND_WITH_SUBBRAND_TOKEN` (currently just
      "Maruti", listed as "Maruti Suzuki <Model>"), skip the sub-brand
      token and take the 3rd word.
    - For every other brand, take the 2nd word.

    This is a heuristic, not a lookup table, so a small number of edge
    cases (e.g. multi-word model names like "Grand i10") will only
    capture the first token of the model name. This is an accepted
    trade-off for cardinality reduction, per project decision.

    Args:
        df: Dataframe containing `name` and `company` columns.

    Returns:
        Copy of df with a new `model` column added.
    """
    df = df.copy()

    def _extract(row: pd.Series) -> str:
        words = str(row[config.COL_NAME]).split()
        company = row[config.COL_COMPANY]
        if company in config.BRAND_WITH_SUBBRAND_TOKEN and len(words) >= 3:
            return words[2]
        return words[1] if len(words) >= 2 else words[0]

    df[config.COL_MODEL] = df.apply(_extract, axis=1)
    return df


# ==========================================================================
# Step 3: Target encoding (leakage-safe, sklearn-compatible)
# ==========================================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target-mean encoder for a single high-cardinality categorical
    column.

    Why target encoding here: `company` (25 categories) and `model`
    (~127 categories) are too high-cardinality for one-hot encoding
    without either exploding dimensionality or discarding information.
    Target encoding maps each category to a (smoothed) average of the
    target variable, which:
      - Keeps the feature space small (1 column per categorical).
      - Captures the fact that, e.g., "Audi" listings command higher
        prices than "Maruti" listings, directly and monotonically.

    Leakage safety: `fit()` must only ever be called on the **training
    split**. At transform time, unseen categories fall back to the
    global training-set mean (via `self.global_mean_`), and smoothing
    pulls rare categories toward the global mean using:

        encoded = (count * category_mean + smoothing * global_mean)
                  / (count + smoothing)

    This prevents rare categories (e.g. a brand with only 2 listings)
    from taking on an extreme, overfit encoding.
    """

    def __init__(self, column: str, smoothing: float = config.TARGET_ENCODING_SMOOTHING) -> None:
        self.column = column
        self.smoothing = smoothing
        self.mapping_: Optional[pd.Series] = None
        self.global_mean_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """Compute smoothed per-category means using training data only."""
        df = pd.DataFrame({self.column: X[self.column].values, "_target": np.asarray(y)})
        self.global_mean_ = float(df["_target"].mean())

        stats = df.groupby(self.column)["_target"].agg(["mean", "count"])
        smoothed = (
            stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
        ) / (stats["count"] + self.smoothing)
        self.mapping_ = smoothed

        logger.info(
            "TargetEncoder fitted on '%s': %d categories, global_mean=%.4f",
            self.column,
            len(self.mapping_),
            self.global_mean_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map categories to their smoothed target means; unseen -> global mean."""
        if self.mapping_ is None or self.global_mean_ is None:
            raise RuntimeError("TargetEncoder must be fit() before transform().")

        X = X.copy()
        encoded_col = f"{self.column}_encoded"
        X[encoded_col] = X[self.column].map(self.mapping_).fillna(self.global_mean_)
        return X


# ==========================================================================
# Step 4: Full pipeline container
# ==========================================================================
@dataclass
class PreprocessingArtifacts:
    """Bundle of fitted transformers + metadata needed to reproduce
    preprocessing at inference time."""

    company_encoder: TargetEncoder
    model_encoder: TargetEncoder
    fuel_type_categories: list = field(default_factory=list)
    feature_columns: list = field(default_factory=list)


def build_feature_matrix(
    df: pd.DataFrame,
    company_encoder: TargetEncoder,
    model_encoder: TargetEncoder,
    fuel_type_categories: list,
) -> pd.DataFrame:
    """
    Assemble the final numeric feature matrix from a cleaned + feature-
    engineered dataframe, using already-fitted encoders.

    This function is called identically during training (after fitting
    encoders on the train split) and during inference (using encoders
    loaded from disk), which guarantees train/serve consistency.

    Args:
        df: Cleaned dataframe with `year`, `kms_driven`, `company`,
            `model`, `fuel_type` columns.
        company_encoder: Fitted TargetEncoder for `company`.
        model_encoder: Fitted TargetEncoder for `model`.
        fuel_type_categories: Full list of fuel type categories seen at
            training time, used to guarantee consistent one-hot columns.

    Returns:
        A numeric-only dataframe ready to feed into LinearRegression.
    """
    df = company_encoder.transform(df)
    df = model_encoder.transform(df)

    fuel_dummies = pd.get_dummies(df[config.COL_FUEL_TYPE], prefix="fuel")
    for category in fuel_type_categories:
        col = f"fuel_{category}"
        if col not in fuel_dummies.columns:
            fuel_dummies[col] = 0
    fuel_dummies = fuel_dummies[[f"fuel_{c}" for c in fuel_type_categories]]

    feature_df = pd.concat(
        [
            df[[config.COL_YEAR, config.COL_KMS_DRIVEN]].reset_index(drop=True),
            df[["company_encoded", "model_encoded"]].reset_index(drop=True),
            fuel_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    return feature_df


def load_and_clean_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Quikr CSV and apply the full raw-cleaning + feature
    engineering sequence (everything except encoding, which must be
    fit on the training split only).

    Args:
        path: Path to raw quikr_car.csv.

    Returns:
        Cleaned dataframe with `model` feature added.
    """
    logger.info("Loading raw data from %s", path)
    raw_df = pd.read_csv(path)

    cleaner = RawDataCleaner()
    cleaned_df = cleaner.clean(raw_df)
    logger.info("Rows dropped per cleaning step: %s", cleaner.rows_dropped_)

    cleaned_df = extract_model_from_name(cleaned_df)
    return cleaned_df


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a simple random train/test split.

    A plain random split (rather than a time-based or stratified split)
    is appropriate here: the dataset has no temporal ordering to
    respect, and the target is continuous.

    Args:
        df: Full cleaned + feature-engineered dataframe.
        test_size: Fraction of rows to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        (train_df, test_df) tuple.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    logger.info("Split data: %d train rows, %d test rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
