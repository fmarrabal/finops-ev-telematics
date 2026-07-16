from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from .metrics import rmse, mape, smape, interval_coverage, mean_interval_width

MODEL_COLUMNS = [
    ("ARIMA(1,1,0) + drift", "Time", "arima"),
    ("Compact Transformer", "Time", "compact_transformer"),
    ("Quadratic demand baseline", "Vehicle", "quadratic_demand_baseline"),
    ("Compact TCN", "Time", "compact_tcn"),
    ("Compact LSTM", "Time", "compact_lstm"),
    ("Probabilistic LSTM", "Time", "probabilistic_lstm"),
    ("Naive last value", "Time", "naive_last_value"),
]


def benchmark_table(ledger: pd.DataFrame) -> pd.DataFrame:
    actual = ledger["actual"].to_numpy(dtype=float)
    rows = []
    for model, inputs, column in MODEL_COLUMNS:
        prediction = ledger[column].to_numpy(dtype=float)
        rows.append(
            {
                "Model": model,
                "Inputs": inputs,
                "RMSE": rmse(actual, prediction),
                "MAPE": mape(actual, prediction),
                "sMAPE": smape(actual, prediction),
            }
        )
    return pd.DataFrame(rows)


def absolute_error_matrix(ledger: pd.DataFrame) -> pd.DataFrame:
    actual = ledger["actual"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            model: np.abs(actual - ledger[column].to_numpy(dtype=float))
            for model, _, column in MODEL_COLUMNS
        },
        index=ledger["forecast_date"],
    )


def comparison_tests(ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    errors = absolute_error_matrix(ledger)
    arrays = [errors[name].to_numpy() for name, _, _ in MODEL_COLUMNS]
    friedman = friedmanchisquare(*arrays)

    arima_name = MODEL_COLUMNS[0][0]
    raw_p = []
    names = []
    statistics = []
    for model, _, _ in MODEL_COLUMNS[1:]:
        test = wilcoxon(
            errors[arima_name].to_numpy(),
            errors[model].to_numpy(),
            alternative="less",
            method="exact",
        )
        names.append(model)
        statistics.append(float(test.statistic))
        raw_p.append(float(test.pvalue))
    reject, adjusted, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    pairwise = pd.DataFrame(
        {
            "Comparator": names,
            "Wilcoxon_statistic": statistics,
            "one_sided_p": raw_p,
            "Holm_adjusted_p": adjusted,
            "Reject_at_0.05": reject,
        }
    )
    summary = {
        "friedman_chi_square": float(friedman.statistic),
        "friedman_degrees_of_freedom": len(MODEL_COLUMNS) - 1,
        "friedman_p_value": float(friedman.pvalue),
        "smallest_holm_adjusted_p": float(np.min(adjusted)),
    }
    return pairwise, summary


def interval_diagnostics(ledger: pd.DataFrame) -> pd.DataFrame:
    actual = ledger["actual"].to_numpy(dtype=float)
    rows = []
    for name, lo_col, hi_col in [
        ("ARIMA(1,1,0) + drift", "arima_lower_95", "arima_upper_95"),
        (
            "Probabilistic LSTM",
            "probabilistic_lstm_lower_95",
            "probabilistic_lstm_upper_95",
        ),
    ]:
        lower = ledger[lo_col].to_numpy(dtype=float)
        upper = ledger[hi_col].to_numpy(dtype=float)
        rows.append(
            {
                "Model": name,
                "Nominal_coverage": 0.95,
                "Empirical_coverage": interval_coverage(actual, lower, upper),
                "Mean_interval_width": mean_interval_width(lower, upper),
            }
        )
    return pd.DataFrame(rows)
