import numpy as np
from finops_repro.data import load_monthly_data
from finops_repro.interrupted_time_series import its_sensitivity_table


def test_primary_interrupted_time_series():
    table = its_sensitivity_table(load_monthly_data()).set_index("Effect_start")
    primary = table.loc["Jul 2021 (primary)"]
    assert abs(primary["Immediate_level_change_percent"] + 45.5976912244) < 1e-8
    assert abs(primary["Level_CI95_lower"] + 53.845) < 0.01
    assert abs(primary["Level_CI95_upper"] + 35.87) < 0.01
    assert abs(primary["Slope_change_percent_per_month"] + 3.5953148737) < 1e-8
    assert abs(primary["Slope_CI95_lower"] + 5.56) < 0.01
    assert abs(primary["Slope_CI95_upper"] + 1.59) < 0.01
    assert abs(primary["R2"] - 0.9744180679) < 1e-10
    assert abs(primary["Pre_slope_percent_per_month"] + 0.9763734337) < 1e-8
    assert abs(primary["Post_slope_percent_per_month"] + 4.5365846081) < 1e-8


def test_lag_sensitivity_display_values():
    table = its_sensitivity_table(load_monthly_data()).set_index("Effect_start")
    expected_level = {
        "Jun 2021": -37.3127790,
        "Jul 2021 (primary)": -45.5976912,
        "Aug 2021": -46.8024559,
        "Sep 2021": -44.4281783,
    }
    expected_slope = {
        "Jun 2021": -3.3198585,
        "Jul 2021 (primary)": -3.5953149,
        "Aug 2021": -2.8171310,
        "Sep 2021": -1.7002023,
    }
    for key in expected_level:
        assert abs(table.loc[key, "Immediate_level_change_percent"] - expected_level[key]) < 1e-6
        assert abs(table.loc[key, "Slope_change_percent_per_month"] - expected_slope[key]) < 1e-6
