from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from sklearn.model_selection import TimeSeriesSplit

from .metrics import regression_metrics, rmse


@dataclass
class QuadraticDemandFit:
    result: object

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(self.result.params, dtype=float)

    def predict(self, vehicles, alpha: float = 0.05) -> pd.DataFrame:
        v = np.asarray(vehicles, dtype=float).reshape(-1)
        design = np.column_stack([np.ones(len(v)), v, v**2])
        frame = self.result.get_prediction(design).summary_frame(alpha=alpha)
        return pd.DataFrame(
            {
                "vehicles_index": v,
                "mean": frame["mean"].to_numpy(),
                "mean_ci_lower": frame["mean_ci_lower"].to_numpy(),
                "mean_ci_upper": frame["mean_ci_upper"].to_numpy(),
                "obs_ci_lower": frame["obs_ci_lower"].to_numpy(),
                "obs_ci_upper": frame["obs_ci_upper"].to_numpy(),
            }
        )


def fit_quadratic_demand(vehicles, cost) -> QuadraticDemandFit:
    v = np.asarray(vehicles, dtype=float)
    y = np.asarray(cost, dtype=float)
    design = np.column_stack([np.ones(len(v)), v, v**2])
    return QuadraticDemandFit(sm.OLS(y, design).fit())


def loocv_quadratic_rmse(vehicles, cost) -> float:
    v = np.asarray(vehicles, dtype=float)
    y = np.asarray(cost, dtype=float)
    predictions = []
    for i in range(len(v)):
        mask = np.arange(len(v)) != i
        fit = fit_quadratic_demand(v[mask], y[mask])
        predictions.append(float(fit.predict([v[i]])["mean"].iloc[0]))
    return rmse(y, predictions)


def jackknife_quadratic(vehicles, cost, final_vehicle: float) -> pd.DataFrame:
    v = np.asarray(vehicles, dtype=float)
    y = np.asarray(cost, dtype=float)
    rows = []
    for i in range(len(v)):
        mask = np.arange(len(v)) != i
        fit = fit_quadratic_demand(v[mask], y[mask])
        beta = fit.coefficients
        final = float(fit.predict([final_vehicle])["mean"].iloc[0])
        rows.append(
            {
                "left_out_index": i,
                "intercept": beta[0],
                "linear": beta[1],
                "quadratic": beta[2],
                "final_counterfactual": final,
            }
        )
    return pd.DataFrame(rows)


def _fit_form(form: str, x: np.ndarray, y: np.ndarray):
    if form == "linear":
        design = np.column_stack([np.ones(len(x)), x])
        result = sm.OLS(y, design).fit()
        predictor = lambda z: np.column_stack([np.ones(len(np.atleast_1d(z))), np.atleast_1d(z)]) @ result.params
    elif form == "quadratic":
        design = np.column_stack([np.ones(len(x)), x, x**2])
        result = sm.OLS(y, design).fit()
        predictor = lambda z: np.column_stack([np.ones(len(np.atleast_1d(z))), np.atleast_1d(z), np.atleast_1d(z) ** 2]) @ result.params
    elif form == "exponential":
        design = np.column_stack([np.ones(len(x)), x])
        result = sm.OLS(np.log(y), design).fit()
        predictor = lambda z: np.exp(np.column_stack([np.ones(len(np.atleast_1d(z))), np.atleast_1d(z)]) @ result.params)
    elif form == "log-linear":
        design = np.column_stack([np.ones(len(x)), np.log(x)])
        result = sm.OLS(y, design).fit()
        predictor = lambda z: np.column_stack([np.ones(len(np.atleast_1d(z))), np.log(np.atleast_1d(z))]) @ result.params
    elif form == "power-law":
        design = np.column_stack([np.ones(len(x)), np.log(x)])
        result = sm.OLS(np.log(y), design).fit()
        predictor = lambda z: np.exp(np.column_stack([np.ones(len(np.atleast_1d(z))), np.log(np.atleast_1d(z))]) @ result.params)
    else:
        raise ValueError(f"Unknown functional form: {form}")
    return result, predictor


def functional_form_sensitivity(
    vehicles,
    cost,
    final_vehicle: float,
    final_observed_cost: float,
) -> pd.DataFrame:
    x = np.asarray(vehicles, dtype=float)
    y = np.asarray(cost, dtype=float)
    rows = []
    labels = {
        "linear": "Linear",
        "quadratic": "Quadratic",
        "exponential": "Exponential",
        "log-linear": "Log-linear",
        "power-law": "Power-law",
    }
    for form in labels:
        result, predictor = _fit_form(form, x, y)
        cv_pred = []
        for i in range(len(x)):
            mask = np.arange(len(x)) != i
            _, fold_predictor = _fit_form(form, x[mask], y[mask])
            cv_pred.append(float(fold_predictor([x[i]])[0]))
        final_cf = float(predictor([final_vehicle])[0])
        gap = 100.0 * (final_cf - final_observed_cost) / final_cf
        rows.append(
            {
                "Form": labels[form],
                "LOOCV_RMSE": rmse(y, cv_pred),
                "Final_CF_cost": final_cf,
                "Gap_percent": gap,
            }
        )
    return pd.DataFrame(rows)


def _quadratic_time_fit(values: np.ndarray):
    t = np.arange(len(values), dtype=float)
    design = np.column_stack([np.ones(len(t)), t, t**2])
    result = sm.OLS(values, design).fit()
    return t, result, design @ result.params


def public_index_baseline_diagnostics(vehicles, cost) -> pd.DataFrame:
    """Recompute diagnostic values from the rounded public index series.

    The manuscript table is separately archived in `published_table3()` because
    several displayed values were generated from the internal full-precision
    export before public two-decimal indexing. Both files are written by the
    pipeline so the distinction is explicit.
    """
    v = np.asarray(vehicles, dtype=float)
    y = np.asarray(cost, dtype=float)
    rows = []
    for label, values in [("Vehicles ~ t^2", v), ("Cloud cost ~ t^2", y)]:
        t, result, pred = _quadratic_time_fit(values)
        fold_rmse = []
        splitter = TimeSeriesSplit(n_splits=4)
        for train, test in splitter.split(t):
            design_train = np.column_stack([np.ones(len(train)), t[train], t[train] ** 2])
            fold = sm.OLS(values[train], design_train).fit()
            design_test = np.column_stack([np.ones(len(test)), t[test], t[test] ** 2])
            fold_rmse.append(rmse(values[test], design_test @ fold.params))
        row = {"Model": label, **regression_metrics(values, pred), "CV_mean_fold_RMSE": float(np.mean(fold_rmse))}
        rows.append(row)
    demand = fit_quadratic_demand(v, y)
    design = np.column_stack([np.ones(len(v)), v, v**2])
    pred = design @ demand.coefficients
    rows.append(
        {
            "Model": "Cloud cost ~ poly(V,2)",
            **regression_metrics(y, pred),
            "CV_mean_fold_RMSE": loocv_quadratic_rmse(v, y),
        }
    )
    return pd.DataFrame(rows)


def published_table3() -> pd.DataFrame:
    """Exact display values reported in manuscript Table 3."""
    return pd.DataFrame(
        [
            ["Vehicles ~ t^2", 0.999, 0.652, 0.525, 0.977, 0.978, 1.766],
            ["Cloud cost ~ t^2", 0.997, 0.986, 0.832, 1.563, 1.560, 3.111],
            ["Cloud cost ~ poly(V,2)", 0.996, 1.222, 1.033, 2.055, 2.053, 1.602],
        ],
        columns=["Model", "R2", "RMSE", "MAE", "MAPE_percent", "sMAPE_percent", "CV"],
    )


def quadratic_planning_projection(values, steps: int = 40, alpha: float = 0.05) -> pd.DataFrame:
    """Quadratic time projection with OLS observation intervals."""
    y = np.asarray(values, dtype=float)
    t = np.arange(len(y), dtype=float)
    design = np.column_stack([np.ones(len(t)), t, t**2])
    result = sm.OLS(y, design).fit()
    future_t = np.arange(len(y), len(y) + steps, dtype=float)
    future_design = np.column_stack([np.ones(len(future_t)), future_t, future_t**2])
    sf = result.get_prediction(future_design).summary_frame(alpha=alpha)
    return pd.DataFrame(
        {
            "month_index": future_t.astype(int),
            "mean": sf["mean"].to_numpy(),
            "lower": sf["obs_ci_lower"].to_numpy(),
            "upper": sf["obs_ci_upper"].to_numpy(),
        }
    )


def uncertainty_scenarios(
    fit: QuadraticDemandFit,
    reference_vehicles: list[tuple[str, float]],
) -> pd.DataFrame:
    rows = []
    for label, vehicle in reference_vehicles:
        pred = fit.predict([vehicle], alpha=0.05).iloc[0]
        rows.append(
            {
                "Horizon_scenario": label,
                "Vehicle_index": vehicle,
                "Mean_CF_cost": pred["mean"],
                "PI95_lower": pred["obs_ci_lower"],
                "PI95_upper": pred["obs_ci_upper"],
            }
        )
    return pd.DataFrame(rows)
