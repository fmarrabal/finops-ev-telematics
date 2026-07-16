from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
BOUNDARY_DATE = pd.Timestamp("2021-06-01")
PRIMARY_EFFECT_DATE = pd.Timestamp("2021-07-01")


def load_monthly_data(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the 51 unique monthly observations."""
    path = path or DATA_DIR / "indexed_monthly_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date",
        "month_index",
        "phase",
        "cloud_cost_index",
        "vehicles_index",
        "total_messages_index",
        "cost_per_vehicle_index",
        "cost_per_billion_messages_index",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if len(df) != 51 or df["date"].nunique() != 51:
        raise ValueError("The full analysis must contain 51 unique monthly dates.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Monthly observations must be sorted chronologically.")
    if (df["cloud_cost_index"] <= 0).any() or (df["vehicles_index"] <= 0).any():
        raise ValueError("Cost and vehicle indices must be strictly positive.")
    boundary = df.loc[df["date"] == BOUNDARY_DATE]
    if len(boundary) != 1:
        raise ValueError("June 2021 must occur exactly once in the unique-month file.")
    return df


def calibration_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return July 2020 through the June 2021 boundary (12 rows)."""
    out = df.loc[df["date"] <= BOUNDARY_DATE].copy()
    if len(out) != 12:
        raise ValueError("Expected 12 calibration observations.")
    return out


def strict_post_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return July 2021 through September 2024 (39 rows)."""
    out = df.loc[df["date"] > BOUNDARY_DATE].copy()
    if len(out) != 39:
        raise ValueError("Expected 39 strictly post-boundary observations.")
    return out


def load_implementation_series(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "implementation_series.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    if len(df) != 40 or df.iloc[0]["date"] != BOUNDARY_DATE:
        raise ValueError("Implementation series must contain 40 rows beginning in June 2021.")
    return df


def load_service_shares(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "service_cost_shares.csv"
    df = pd.read_csv(path)
    if abs(df["share_percent"].sum() - 100.1) > 1e-9:
        # Values are rounded to one decimal, so they sum to 100.1 rather than 100.0.
        raise ValueError("Unexpected service-share total.")
    return df


def load_prediction_ledger(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "published_common_fold_predictions.csv"
    df = pd.read_csv(path, parse_dates=["forecast_date"])
    if len(df) != 6:
        raise ValueError("The common-fold ledger must contain six held-out months.")
    return df


def load_long_horizon_scenarios(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "long_horizon_scenarios.csv"
    df = pd.read_csv(path)
    if len(df) != 40:
        raise ValueError("Expected a 40-month descriptive scenario ledger.")
    return df


def load_reference_results(path: Path | None = None) -> dict:
    path = path or DATA_DIR / "manuscript_reference_results.json"
    return json.loads(path.read_text(encoding="utf-8"))
