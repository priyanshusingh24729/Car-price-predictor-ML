"""
evaluate.py
===========
Evaluates a trained model on the held-out test set and produces the
standard regression diagnostic plots:
  - Actual vs Predicted scatter
  - Residual plot
  - Error distribution histogram

All metrics are reported on the ORIGINAL Price scale (rupees), not the
log scale the model was trained on -- log-scale RMSE isn't meaningful
to a business stakeholder, so predictions are exponentiated back
(`np.expm1`) before computing MAE/MSE/RMSE/R^2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics on the original (non-log) scale.

    Args:
        y_true: Ground-truth prices (rupees).
        y_pred: Predicted prices (rupees).

    Returns:
        Dict with keys: mae, mse, rmse, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    logger.info(
        "Evaluation metrics -- MAE: %.2f | MSE: %.2f | RMSE: %.2f | R2: %.4f",
        mae,
        mse,
        rmse,
        r2,
    )
    return metrics


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None
) -> None:
    """Scatter plot of predicted vs actual prices with a perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Actual vs Predicted Price")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        logger.info("Saved actual-vs-predicted plot to %s", save_path)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None
) -> None:
    """Residual plot: predicted price vs (actual - predicted)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, color="darkorange", edgecolor="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Price (₹)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        logger.info("Saved residual plot to %s", save_path)
    plt.close(fig)


def plot_error_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None
) -> None:
    """Histogram of prediction errors (residuals)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(residuals, bins=30, color="seagreen", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction Error (₹)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        logger.info("Saved error distribution plot to %s", save_path)
    plt.close(fig)


def evaluate_model(
    model, X_test: pd.DataFrame, y_test_log: pd.Series, output_dir: Optional[Path] = None
) -> dict:
    """
    Full evaluation entry point: predicts on X_test, converts back from
    log-scale, computes metrics, and generates all diagnostic plots.

    Args:
        model: Fitted sklearn regressor (trained on log-target).
        X_test: Test feature matrix.
        y_test_log: True log-transformed test targets.
        output_dir: If provided, plots are saved here as PNGs.

    Returns:
        Dict of computed metrics.
    """
    y_pred_log = model.predict(X_test)

    # Convert back to original rupee scale for interpretable metrics
    y_true = np.expm1(y_test_log.values if hasattr(y_test_log, "values") else y_test_log)
    y_pred = np.expm1(y_pred_log)

    metrics = compute_metrics(y_true, y_pred)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_actual_vs_predicted(y_true, y_pred, output_dir / "actual_vs_predicted.png")
        plot_residuals(y_true, y_pred, output_dir / "residual_plot.png")
        plot_error_distribution(y_true, y_pred, output_dir / "error_distribution.png")

    return metrics
