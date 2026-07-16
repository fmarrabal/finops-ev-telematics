import pandas as pd
from finops_repro.data import (
    load_monthly_data,
    calibration_data,
    strict_post_data,
    load_implementation_series,
)


def test_boundary_is_counted_once():
    df = load_monthly_data()
    assert len(df) == 51
    assert df["date"].nunique() == 51
    assert (df["date"] == pd.Timestamp("2021-06-01")).sum() == 1
    assert len(calibration_data(df)) == 12
    assert len(strict_post_data(df)) == 39
    assert len(load_implementation_series()) == 40
