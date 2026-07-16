from __future__ import annotations

import numpy as np


def _arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual={a.shape}, predicted={p.shape}")
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite values.")
    return a, p


def rmse(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def mape(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero.")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def smape(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    denom = np.abs(a) + np.abs(p)
    if np.any(denom == 0):
        raise ValueError("sMAPE is undefined when actual and predicted are both zero.")
    return float(np.mean(200.0 * np.abs(a - p) / denom))


def r2(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(actual, predicted) -> dict[str, float]:
    return {
        "R2": r2(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MAPE_percent": mape(actual, predicted),
        "sMAPE_percent": smape(actual, predicted),
    }


def interval_coverage(actual, lower, upper) -> float:
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("Actual, lower, and upper arrays must have the same shape.")
    return float(np.mean((a >= lo) & (a <= hi)))


def mean_interval_width(lower, upper) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean(hi - lo))
