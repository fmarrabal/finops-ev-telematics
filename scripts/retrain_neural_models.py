#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finops_repro.data import load_monthly_data, calibration_data, load_prediction_ledger
from finops_repro.metrics import rmse, mape, smape
from finops_repro.neural import TrainConfig, expanding_window_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain compact PyTorch forecasting models on the six common folds.")
    parser.add_argument("--output-dir", default="outputs/retrained_neural")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--mc-draws", type=int, default=200)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration = calibration_data(load_monthly_data())
    actual = calibration["cloud_cost_index"].to_numpy(float)[6:]
    predictions = expanding_window_predictions(
        calibration["cloud_cost_index"].to_numpy(float),
        config=TrainConfig(epochs=args.epochs, patience=args.patience, mc_draws=args.mc_draws),
    )
    dates = calibration["date"].iloc[6:].reset_index(drop=True)
    frame = pd.DataFrame({"forecast_date": dates, "actual": actual})
    rows = []
    for family, values in predictions.items():
        frame[family] = values["point"]
        frame[f"{family}_lower_95"] = values["lower"]
        frame[f"{family}_upper_95"] = values["upper"]
        rows.append(
            {
                "Model": family,
                "RMSE": rmse(actual, values["point"]),
                "MAPE": mape(actual, values["point"]),
                "sMAPE": smape(actual, values["point"]),
            }
        )
    frame.to_csv(output / "retrained_predictions.csv", index=False, float_format="%.10f")
    pd.DataFrame(rows).to_csv(output / "retrained_metrics.csv", index=False, float_format="%.10f")

    # Side-by-side comparison to the immutable manuscript ledger.
    published = load_prediction_ledger()
    comparison = []
    mapping = {
        "transformer": "compact_transformer",
        "tcn": "compact_tcn",
        "lstm": "compact_lstm",
        "probabilistic_lstm": "probabilistic_lstm",
    }
    for family, column in mapping.items():
        comparison.append(
            {
                "Model": family,
                "RMSE_retrained_vs_published_predictions": rmse(
                    published[column].to_numpy(float), predictions[family]["point"]
                ),
            }
        )
    pd.DataFrame(comparison).to_csv(output / "comparison_to_published_ledger.csv", index=False)
    print(f"Neural retraining outputs written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
