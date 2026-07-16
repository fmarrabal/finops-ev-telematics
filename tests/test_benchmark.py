import numpy as np
from finops_repro.data import load_prediction_ledger
from finops_repro.benchmark import benchmark_table, comparison_tests, interval_diagnostics


def test_common_fold_metrics_match_manuscript():
    table = benchmark_table(load_prediction_ledger()).set_index("Model")
    expected = {
        "ARIMA(1,1,0) + drift": (2.3952486222, 2.8749684486, 2.9244966933),
        "Compact Transformer": (2.65, 3.12, 3.16),
        "Quadratic demand baseline": (3.4198105099, 3.7331710907, 3.6240653257),
        "Compact TCN": (5.62, 7.24, 7.53),
        "Compact LSTM": (7.0, 8.01, 8.37),
        "Probabilistic LSTM": (7.65, 8.19, 8.60),
        "Naive last value": (8.1280296915, 10.1159081885, 10.6651945125),
    }
    for name, target in expected.items():
        values = table.loc[name, ["RMSE", "MAPE", "sMAPE"]].to_numpy(float)
        np.testing.assert_allclose(values, target, atol=5e-8, rtol=0)


def test_friedman_and_holm_results():
    pairwise, summary = comparison_tests(load_prediction_ledger())
    assert abs(summary["friedman_chi_square"] - 20.5) < 1e-12
    assert abs(summary["friedman_p_value"] - 0.0022551456011152115) < 1e-15
    assert abs(summary["smallest_holm_adjusted_p"] - 0.09375) < 1e-12
    assert not pairwise["Reject_at_0.05"].any()


def test_interval_diagnostics():
    table = interval_diagnostics(load_prediction_ledger()).set_index("Model")
    assert table.loc["ARIMA(1,1,0) + drift", "Empirical_coverage"] == 1.0
    assert abs(table.loc["ARIMA(1,1,0) + drift", "Mean_interval_width"] - 7.23065771758) < 1e-9
    assert table.loc["Probabilistic LSTM", "Empirical_coverage"] == 1.0
    assert abs(table.loc["Probabilistic LSTM", "Mean_interval_width"] - 23.45) < 1e-12
