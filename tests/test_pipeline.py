from finops_repro.pipeline import reproduce


def test_numerical_pipeline(tmp_path):
    summary = reproduce(tmp_path / "out", make_figures=False)
    assert summary["all_reference_checks_passed"] is True
    assert (tmp_path / "out" / "results_summary.json").exists()
    assert (tmp_path / "out" / "tables" / "table6_interrupted_time_series.csv").exists()
