"""
main.py
=======
Command-line entry point that orchestrates the Car Price Prediction
pipeline end-to-end.

Usage:
    python main.py --stage train      # clean data, fit encoders + model, save artifacts
    python main.py --stage evaluate   # load/train model, score on test set, save plots
    python main.py --stage all        # train + evaluate (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_model
from src.train import train_model
from src.utils import get_logger

logger = get_logger(__name__)

PLOTS_OUTPUT_DIR = Path("notebooks/eda_plots")


def run_train() -> dict:
    logger.info("=== STAGE: TRAIN ===")
    return train_model()


def run_evaluate(results: dict) -> None:
    logger.info("=== STAGE: EVALUATE ===")
    metrics = evaluate_model(
        results["model"],
        results["X_test"],
        results["y_test_log"],
        output_dir=PLOTS_OUTPUT_DIR,
    )
    print("\n" + "=" * 50)
    print("FINAL TEST SET METRICS")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key.upper():>6}: {value:,.4f}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Car Price Prediction Pipeline")
    parser.add_argument(
        "--stage",
        choices=["train", "evaluate", "all"],
        default="all",
        help="Which pipeline stage to run.",
    )
    args = parser.parse_args()

    if args.stage in ("train", "all"):
        results = run_train()
    if args.stage == "evaluate":
        # If only evaluating, we still need a freshly trained model in memory
        # since predictions/artifacts aren't cached between CLI invocations.
        results = run_train()
    if args.stage in ("evaluate", "all"):
        run_evaluate(results)


if __name__ == "__main__":
    main()
