from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _percent(log_coefficient: float) -> float:
    return float((np.exp(log_coefficient) - 1.0) * 100.0)


def fit_segmented_its(
    data: pd.DataFrame,
    effect_start: str | pd.Timestamp,
    maxlags: int = 3,
):
    """Fit the segmented log-unit-cost model with corrected Newey-West SEs."""
    start = pd.Timestamp(effect_start)
    dates = pd.to_datetime(data["date"])
    matches = np.flatnonzero(dates.to_numpy() == start.to_datetime64())
    if len(matches) != 1:
        raise ValueError(f"Effect start {start.date()} is not uniquely present.")
    start_index = int(matches[0])
    t = np.arange(len(data), dtype=float)
    intervention = (t >= start_index).astype(float)
    post_time = (t - start_index) * intervention
    design = np.column_stack([np.ones(len(t)), t, intervention, post_time])
    unit_cost = 100.0 * data["cloud_cost_index"].to_numpy(dtype=float) / data[
        "vehicles_index"
    ].to_numpy(dtype=float)
    result = sm.OLS(np.log(unit_cost), design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )
    return result, unit_cost, design


def its_sensitivity_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, date in [
        ("Jun 2021", "2021-06-01"),
        ("Jul 2021 (primary)", "2021-07-01"),
        ("Aug 2021", "2021-08-01"),
        ("Sep 2021", "2021-09-01"),
    ]:
        result, _, _ = fit_segmented_its(data, date)
        ci = result.conf_int(alpha=0.05)
        rows.append(
            {
                "Effect_start": label,
                "Immediate_level_change_percent": _percent(result.params[2]),
                "Level_CI95_lower": _percent(ci[2, 0]),
                "Level_CI95_upper": _percent(ci[2, 1]),
                "Slope_change_percent_per_month": _percent(result.params[3]),
                "Slope_CI95_lower": _percent(ci[3, 0]),
                "Slope_CI95_upper": _percent(ci[3, 1]),
                "R2": float(result.rsquared),
                "Level_p_value": float(result.pvalues[2]),
                "Slope_p_value": float(result.pvalues[3]),
                "Pre_slope_percent_per_month": _percent(result.params[1]),
                "Pre_slope_p_value": float(result.pvalues[1]),
                "Post_slope_percent_per_month": _percent(
                    result.params[1] + result.params[3]
                ),
            }
        )
    return pd.DataFrame(rows)


def fitted_primary_series(data: pd.DataFrame) -> pd.DataFrame:
    result, unit_cost, design = fit_segmented_its(data, "2021-07-01")
    fitted = np.exp(design @ result.params)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"]),
            "observed_cost_per_vehicle": unit_cost,
            "segmented_fit": fitted,
        }
    )
