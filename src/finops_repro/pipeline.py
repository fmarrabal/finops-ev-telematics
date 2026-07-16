from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import benchmark_table, comparison_tests, interval_diagnostics
from .counterfactual import (
    fit_quadratic_demand,
    functional_form_sensitivity,
    jackknife_quadratic,
    loocv_quadratic_rmse,
    public_index_baseline_diagnostics,
    published_table3,
    uncertainty_scenarios,
)
from .data import (
    DATA_DIR,
    calibration_data,
    load_implementation_series,
    load_long_horizon_scenarios,
    load_monthly_data,
    load_prediction_ledger,
    load_reference_results,
    load_service_shares,
)
from .interrupted_time_series import fitted_primary_series, its_sensitivity_table
from .plots import plot_figure1, plot_figure2, plot_figure3, plot_figure4, plot_figure5


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def _jsonable(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _kpi_table(implementation: pd.DataFrame) -> pd.DataFrame:
    first = implementation.iloc[0]
    final = implementation.iloc[-1]
    rows = [
        ("Vehicle Fleet (indexed)", first["vehicles_index"], final["vehicles_index"]),
        ("Monthly Messages (indexed)", first["total_messages_index"], final["total_messages_index"]),
        ("Total Cloud Cost (indexed)", first["cloud_cost_index"], final["cloud_cost_index"]),
        (
            "Cost per Billion Messages",
            first["cost_per_billion_messages_index"],
            final["cost_per_billion_messages_index"],
        ),
        ("Cost per Active Vehicle", first["cost_per_vehicle_index"], final["cost_per_vehicle_index"]),
    ]
    out = []
    for metric, pre, post in rows:
        out.append(
            {
                "Metric": metric,
                "Pre_FinOps": float(pre),
                "Post_FinOps": float(post),
                "Change_percent": float((post / pre - 1.0) * 100.0),
            }
        )
    out.insert(
        4,
        {
            "Metric": "Efficiency Factor",
            "Pre_FinOps": 1.0,
            "Post_FinOps": float(
                first["cost_per_billion_messages_index"]
                / final["cost_per_billion_messages_index"]
            ),
            "Change_percent": float(
                (
                    first["cost_per_billion_messages_index"]
                    / final["cost_per_billion_messages_index"]
                    - 1.0
                )
                * 100.0
            ),
        },
    )
    return pd.DataFrame(out)


def _counterfactual_monthly(
    fit,
    implementation: pd.DataFrame,
) -> pd.DataFrame:
    vehicles = implementation["vehicles_index"].to_numpy(float)
    result = implementation[["date", "implementation_month", "vehicles_index", "cloud_cost_index"]].copy()
    for label, alpha in [("50", 0.50), ("80", 0.20), ("95", 0.05)]:
        pred = fit.predict(vehicles, alpha=alpha)
        if label == "95":
            result["counterfactual_mean"] = pred["mean"]
        result[f"PI{label}_lower"] = pred["obs_ci_lower"]
        result[f"PI{label}_upper"] = pred["obs_ci_upper"]
    result["counterfactual_gap"] = result["counterfactual_mean"] - result["cloud_cost_index"]
    result["counterfactual_gap_percent"] = (
        100.0 * result["counterfactual_gap"] / result["counterfactual_mean"]
    )
    return result


def _check_close(name: str, actual: float, expected: float, atol: float, rows: list[dict]) -> None:
    passed = bool(abs(actual - expected) <= atol)
    rows.append(
        {
            "Check": name,
            "Actual": actual,
            "Expected": expected,
            "Absolute_tolerance": atol,
            "Passed": passed,
        }
    )


def reproduce(output_dir: str | Path = "outputs", make_figures: bool = True) -> dict:
    """Run the complete deterministic R2 analysis and write all artifacts."""
    output_dir = Path(output_dir).resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly_data()
    calibration = calibration_data(monthly)
    implementation = load_implementation_series()
    ledger = load_prediction_ledger()
    shares = load_service_shares()
    long_scenarios = load_long_horizon_scenarios()
    reference = load_reference_results()

    vehicles = calibration["vehicles_index"].to_numpy(float)
    cost = calibration["cloud_cost_index"].to_numpy(float)
    fit = fit_quadratic_demand(vehicles, cost)
    final_vehicle = float(implementation.iloc[-1]["vehicles_index"])
    final_observed = float(implementation.iloc[-1]["cloud_cost_index"])

    table3 = published_table3()
    diagnostics_public = public_index_baseline_diagnostics(vehicles, cost)
    table4 = functional_form_sensitivity(vehicles, cost, final_vehicle, final_observed)
    table5 = benchmark_table(ledger)
    pairwise, benchmark_summary = comparison_tests(ledger)
    interval_table = interval_diagnostics(ledger)
    table6 = its_sensitivity_table(monthly)
    table7 = _kpi_table(implementation)
    table8 = shares.rename(columns={"cost_item": "Cost_item", "share_percent": "Share_percent"})

    reference_scenarios = [
        ("6 months - reference", 135.86),
        ("12 months - reference", 171.23),
        ("24 months - reference", 275.07),
        ("39 months - reference", final_vehicle),
        ("Final - low (-15%)", final_vehicle * 0.85),
        ("Final - reference", final_vehicle),
        ("Final - high (+15%)", final_vehicle * 1.15),
    ]
    table9 = uncertainty_scenarios(fit, reference_scenarios)
    # Display the scenario workload indices at the same precision used in the manuscript,
    # while retaining the unrounded values for the cost and interval calculations.
    table9["Vehicle_index"] = table9["Vehicle_index"].round(2)
    jackknife = jackknife_quadratic(vehicles, cost, final_vehicle)
    cf_monthly = _counterfactual_monthly(fit, implementation)
    its_fitted = fitted_primary_series(monthly)

    _write_csv(table3, table_dir / "table3_baseline_diagnostics_published.csv")
    _write_csv(diagnostics_public, table_dir / "table3_diagnostics_recomputed_public_indices.csv")
    _write_csv(table4, table_dir / "table4_functional_form_sensitivity.csv")
    _write_csv(table5, table_dir / "table5_common_fold_benchmark.csv")
    _write_csv(ledger, table_dir / "table5_prediction_ledger.csv")
    _write_csv(pairwise, table_dir / "table5_pairwise_holm_tests.csv")
    _write_csv(interval_table, table_dir / "table5_interval_diagnostics.csv")
    _write_csv(table6, table_dir / "table6_interrupted_time_series.csv")
    _write_csv(table7, table_dir / "table7_observed_kpis.csv")
    _write_csv(table8, table_dir / "table8_service_cost_shares.csv")
    _write_csv(table9, table_dir / "table9_uncertainty_scenarios.csv")
    _write_csv(jackknife, table_dir / "quadratic_jackknife.csv")
    _write_csv(cf_monthly, table_dir / "monthly_counterfactual_and_intervals.csv")
    _write_csv(its_fitted, table_dir / "interrupted_time_series_fitted.csv")

    final_prediction = fit.predict([final_vehicle], alpha=0.05).iloc[0]
    summary = {
        "data": {
            "unique_months": len(monthly),
            "calibration_months": len(calibration),
            "strict_post_months": int((monthly["date"] > pd.Timestamp("2021-06-01")).sum()),
            "implementation_series_months": len(implementation),
        },
        "quadratic_counterfactual": {
            "coefficients": fit.coefficients,
            "loocv_rmse": loocv_quadratic_rmse(vehicles, cost),
            "final_vehicle_index": final_vehicle,
            "final_observed_cost": final_observed,
            "final_counterfactual_mean": float(final_prediction["mean"]),
            "final_pi95_lower": float(final_prediction["obs_ci_lower"]),
            "final_pi95_upper": float(final_prediction["obs_ci_upper"]),
            "final_gap_percent": float(
                100.0 * (final_prediction["mean"] - final_observed) / final_prediction["mean"]
            ),
            "jackknife_gamma_min": float(jackknife["quadratic"].min()),
            "jackknife_gamma_max": float(jackknife["quadratic"].max()),
            "jackknife_final_min": float(jackknife["final_counterfactual"].min()),
            "jackknife_final_max": float(jackknife["final_counterfactual"].max()),
        },
        "benchmark": benchmark_summary,
        "primary_its": table6.loc[table6["Effect_start"] == "Jul 2021 (primary)"].iloc[0].to_dict(),
        "observed_kpis": table7.to_dict(orient="records"),
    }

    checks: list[dict] = []
    ref_cf = reference["counterfactual"]
    _check_close("Quadratic intercept", fit.coefficients[0], ref_cf["quadratic_coefficients"][0], 1e-5, checks)
    _check_close("Quadratic linear coefficient", fit.coefficients[1], ref_cf["quadratic_coefficients"][1], 1e-7, checks)
    _check_close("Quadratic curvature", fit.coefficients[2], ref_cf["quadratic_coefficients"][2], 1e-9, checks)
    _check_close("Quadratic LOOCV RMSE", summary["quadratic_counterfactual"]["loocv_rmse"], ref_cf["loocv_rmse"], 1e-6, checks)
    _check_close("Final counterfactual", summary["quadratic_counterfactual"]["final_counterfactual_mean"], ref_cf["final_cost"], 1e-5, checks)
    _check_close("Friedman chi-square", benchmark_summary["friedman_chi_square"], reference["benchmark"]["friedman_chi2"], 1e-12, checks)
    _check_close("Smallest Holm-adjusted p", benchmark_summary["smallest_holm_adjusted_p"], reference["benchmark"]["smallest_holm_adjusted_p"], 1e-12, checks)
    primary = summary["primary_its"]
    _check_close("Primary ITS level effect", primary["Immediate_level_change_percent"], reference["its_primary"]["level_percent"], 1e-6, checks)
    _check_close("Primary ITS slope change", primary["Slope_change_percent_per_month"], reference["its_primary"]["slope_percent"], 1e-6, checks)
    check_df = pd.DataFrame(checks)
    _write_csv(check_df, table_dir / "reproducibility_checks.csv")
    summary["all_reference_checks_passed"] = bool(check_df["Passed"].all())

    if make_figures:
        plot_figure1(figure_dir / "Figure1_evaluation_architecture.png")
        plot_figure2(calibration, figure_dir / "Figure2_planning_projections.png")
        plot_figure3(calibration, figure_dir / "Figure3_cost_vehicle_quadratic.png")
        plot_figure4(
            monthly,
            calibration,
            implementation,
            long_scenarios,
            figure_dir / "Figure4_forecasting_and_finops_impact.png",
        )
        plot_figure5(
            monthly,
            calibration,
            implementation,
            figure_dir / "Figure5_its_and_uncertainty.png",
        )

    (output_dir / "results_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Reproducibility report",
        "",
        f"All reference checks passed: **{summary['all_reference_checks_passed']}**",
        "",
        "## Key reproduced results",
        "",
        f"- 51 unique monthly observations; 12 calibration months and 39 strictly post-boundary months.",
        f"- Quadratic counterfactual: C(V) = {fit.coefficients[0]:.5f} {fit.coefficients[1]:+.6f}V {fit.coefficients[2]:+.8f}V^2.",
        f"- Final quadratic no-FinOps estimate: {final_prediction['mean']:.2f} (95% PI {final_prediction['obs_ci_lower']:.2f}-{final_prediction['obs_ci_upper']:.2f}) versus observed {final_observed:.2f}.",
        f"- Friedman chi-square({benchmark_summary['friedman_degrees_of_freedom']}) = {benchmark_summary['friedman_chi_square']:.2f}, p = {benchmark_summary['friedman_p_value']:.4f}; smallest Holm-adjusted p = {benchmark_summary['smallest_holm_adjusted_p']:.3f}.",
        f"- Primary ITS immediate level change: {primary['Immediate_level_change_percent']:.1f}%; slope change: {primary['Slope_change_percent_per_month']:.2f}% per month.",
        "",
        "## Notes",
        "",
        "The exact common-fold forecast ledger is archived in `data/published_common_fold_predictions.csv` so the manuscript comparison remains deterministic across deep-learning backends. Executable PyTorch implementations and an optional retraining command are included separately.",
        "",
        "The public indexed observations are rounded. For transparency, the pipeline writes both the exact display values used in manuscript Table 3 and a fresh diagnostic recomputation from the public index CSV.",
    ]
    (output_dir / "REPRODUCIBILITY_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary
