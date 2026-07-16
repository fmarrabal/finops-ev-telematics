import numpy as np
from finops_repro.data import load_monthly_data, calibration_data, load_implementation_series
from finops_repro.counterfactual import (
    fit_quadratic_demand,
    loocv_quadratic_rmse,
    functional_form_sensitivity,
    jackknife_quadratic,
)


def test_quadratic_coefficients_and_final_projection():
    pre = calibration_data(load_monthly_data())
    fit = fit_quadratic_demand(pre["vehicles_index"], pre["cloud_cost_index"])
    np.testing.assert_allclose(
        fit.coefficients,
        [39.640489480822, -0.33352844204562904, 0.009421691783910951],
        rtol=0,
        atol=1e-8,
    )
    assert abs(loocv_quadratic_rmse(pre["vehicles_index"], pre["cloud_cost_index"]) - 1.6003925587) < 1e-9
    pred = fit.predict([440.14], alpha=0.05).iloc[0]
    assert abs(pred["mean"] - 1718.04174745696) < 1e-8
    assert abs(pred["obs_ci_lower"] - 1366.99476226626) < 1e-8
    assert abs(pred["obs_ci_upper"] - 2069.08873264765) < 1e-8


def test_functional_form_sensitivity():
    pre = calibration_data(load_monthly_data())
    final = load_implementation_series().iloc[-1]
    table = functional_form_sensitivity(
        pre["vehicles_index"],
        pre["cloud_cost_index"],
        float(final["vehicles_index"]),
        float(final["cloud_cost_index"]),
    ).set_index("Form")
    expected = {
        "Linear": (4.7045201653, 406.3159497727, 88.9051857243),
        "Quadratic": (1.6003925587, 1718.0417474570, 97.3760823876),
        "Exponential": (2.0766187420, 14341.9874522069, 99.6856781520),
        "Log-linear": (7.9280994728, 169.6220845238, 73.4232720188),
        "Power-law": (5.3330624662, 337.9219331465, 86.6596407104),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(table.loc[name].to_numpy(float), values, atol=1e-8, rtol=0)


def test_jackknife_ranges():
    pre = calibration_data(load_monthly_data())
    result = jackknife_quadratic(pre["vehicles_index"], pre["cloud_cost_index"], 440.14)
    assert abs(result["quadratic"].min() - 0.00873234917148) < 1e-12
    assert abs(result["quadratic"].max() - 0.01014092953757) < 1e-12
    assert abs(result["final_counterfactual"].min() - 1626.3023022488) < 1e-8
    assert abs(result["final_counterfactual"].max() - 1817.8011583243) < 1e-8


def test_high_demand_scenario_uses_unrounded_15_percent_multiplier():
    pre = calibration_data(load_monthly_data())
    fit = fit_quadratic_demand(pre["vehicles_index"], pre["cloud_cost_index"])
    final_vehicle = float(load_implementation_series().iloc[-1]["vehicles_index"])
    pred = fit.predict([final_vehicle * 1.15]).iloc[0]
    assert abs(pred["mean"] - 2284.649016617397) < 1e-8
    assert abs(pred["obs_ci_lower"] - 1798.357107396551) < 1e-8
    assert abs(pred["obs_ci_upper"] - 2770.9409258382434) < 1e-8
