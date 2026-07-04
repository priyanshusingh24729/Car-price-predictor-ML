"""
predict.py
==========
Inference-time interface for the trained car-price model.

Loads the persisted model + preprocessing artifacts exactly once
(via `CarPricePredictor`) and exposes a simple `predict()` method that
takes raw, human-readable inputs (company, model, year, kms_driven,
fuel_type) and returns a predicted price in rupees.

This is the module the Streamlit frontend (and any other consumer,
e.g. a future REST API) should import -- it never re-implements
feature engineering, guaranteeing consistency with training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import PreprocessingArtifacts, build_feature_matrix
from src.utils import get_logger, load_pickle

logger = get_logger(__name__)


class InvalidInputError(Exception):
    """Raised when prediction input fails validation."""


@dataclass
class PredictionResult:
    """Structured result returned by `CarPricePredictor.predict`."""

    predicted_price: float
    input_summary: dict


class CarPricePredictor:
    """
    Loads trained artifacts once and serves predictions for new cars.

    Usage:
        predictor = CarPricePredictor()
        result = predictor.predict(
            company="Maruti", model="Swift", year=2015,
            kms_driven=35000, fuel_type="Petrol"
        )
        print(result.predicted_price)
    """

    def __init__(self) -> None:
        self.model = load_pickle(config.MODEL_PATH)
        self.artifacts: PreprocessingArtifacts = load_pickle(config.FEATURE_METADATA_PATH)
        logger.info("Loaded model and preprocessing artifacts for inference.")

    def _validate_input(
        self, company: str, model: str, year: int, kms_driven: int, fuel_type: str
    ) -> None:
        if not company or not isinstance(company, str):
            raise InvalidInputError("Company must be a non-empty string.")
        if not model or not isinstance(model, str):
            raise InvalidInputError("Model must be a non-empty string.")
        if not (config.MIN_VALID_YEAR <= year <= config.CURRENT_YEAR):
            raise InvalidInputError(
                f"Year must be between {config.MIN_VALID_YEAR} and {config.CURRENT_YEAR}."
            )
        if not (0 <= kms_driven <= config.MAX_VALID_KMS):
            raise InvalidInputError(f"kms_driven must be between 0 and {config.MAX_VALID_KMS}.")
        if fuel_type not in self.artifacts.fuel_type_categories:
            raise InvalidInputError(
                f"fuel_type must be one of {self.artifacts.fuel_type_categories}."
            )

    def predict(
        self, company: str, model: str, year: int, kms_driven: int, fuel_type: str
    ) -> PredictionResult:
        """
        Predict the resale price for a single car.

        Args:
            company: Car manufacturer/brand (e.g. "Maruti", "Hyundai").
            model: Car model name (e.g. "Swift", "i20").
            year: Manufacturing year.
            kms_driven: Total kilometers driven.
            fuel_type: One of the fuel types seen at training time
                (e.g. "Petrol", "Diesel").

        Returns:
            PredictionResult with the predicted price in rupees.

        Raises:
            InvalidInputError: If any input fails validation.
        """
        self._validate_input(company, model, year, kms_driven, fuel_type)

        input_df = pd.DataFrame(
            [
                {
                    config.COL_COMPANY: company,
                    config.COL_MODEL: model,
                    config.COL_YEAR: year,
                    config.COL_KMS_DRIVEN: kms_driven,
                    config.COL_FUEL_TYPE: fuel_type,
                }
            ]
        )

        feature_df = build_feature_matrix(
            input_df,
            self.artifacts.company_encoder,
            self.artifacts.model_encoder,
            self.artifacts.fuel_type_categories,
        )
        # Ensure column order matches training exactly
        feature_df = feature_df[self.artifacts.feature_columns]

        pred_log = self.model.predict(feature_df)[0]
        predicted_price = float(np.expm1(pred_log))

        logger.info(
            "Predicted price for %s %s (%d, %d km, %s): ₹%.2f",
            company,
            model,
            year,
            kms_driven,
            fuel_type,
            predicted_price,
        )

        return PredictionResult(
            predicted_price=predicted_price,
            input_summary={
                "company": company,
                "model": model,
                "year": year,
                "kms_driven": kms_driven,
                "fuel_type": fuel_type,
            },
        )
