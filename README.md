# FinOps-Driven Cloud Cost Optimization in EV Telematics

Reproducibility package for the manuscript:

> **FinOps-Driven Cloud Cost Optimization in Electric Vehicle Telematics Platforms: A Longitudinal Case Study with Demand-Controlled Forecasting**  
> Víctor Valdivieso, Christian Sonderstrup, Dora Cama-Pinto, Alejandro Cama-Pinto, and Francisco Manuel Arrabal-Campos  
> *Expert Systems with Applications*, manuscript ESWA-D-26-06640R1/R2.

The repository reproduces the numerical analyses, statistical tests, manuscript tables, and Figures 1-5 from the indexed case-study data. Raw monetary, billing, and vehicle payload data remain proprietary and are not distributed.

## Main reproduced results

- 51 unique monthly observations: 12 calibration months, one shared June 2021 boundary counted once, and 39 strictly post-boundary months.
- Observed fleet growth: 100 to 440.14; message growth: 100 to 1006.85.
- Observed total cloud cost: 100 to 45.08.
- Cost per active vehicle: 100 to 10.24; cost per billion messages: 100 to 4.48.
- Quadratic demand counterfactual at the final vehicle index: 1718.04, with 95% prediction interval 1366.99-2069.09.
- Common-fold benchmark: ARIMA RMSE 2.39; compact Transformer RMSE 2.65.
- Friedman chi-square(6) = 20.50, p = 0.0023; no pairwise comparison with ARIMA survives Holm correction.
- Primary interrupted time-series estimate: -45.6% immediate level change and -3.60% additional monthly slope change in cost per active vehicle.

## Quick reproduction

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduce.py
pytest -q
```

For the exact dependency versions used to validate this archive, replace `requirements.txt` with `requirements-lock.txt`. Development and test pins are available in `requirements-dev-lock.txt`.

The command writes:

```text
outputs/
├── REPRODUCIBILITY_REPORT.md
├── results_summary.json
├── figures/
│   ├── Figure1_evaluation_architecture.png
│   ├── Figure2_planning_projections.png
│   ├── Figure3_cost_vehicle_quadratic.png
│   ├── Figure4_forecasting_and_finops_impact.png
│   └── Figure5_its_and_uncertainty.png
└── tables/
    ├── table3_baseline_diagnostics_published.csv
    ├── table4_functional_form_sensitivity.csv
    ├── table5_common_fold_benchmark.csv
    ├── table5_pairwise_holm_tests.csv
    ├── table6_interrupted_time_series.csv
    ├── table7_observed_kpis.csv
    ├── table8_service_cost_shares.csv
    ├── table9_uncertainty_scenarios.csv
    └── ...
```

A numerical-only run is available through:

```bash
python reproduce.py --skip-figures
```

or:

```bash
make numerical
```

## Why an exact prediction ledger is included

Deep-learning forecasts can change slightly across PyTorch, BLAS, compiler, and CPU versions even when random seeds are fixed. To make the published statistical comparison completely deterministic, `data/published_common_fold_predictions.csv` stores the six held-out forecasts used in manuscript Table 5. The pipeline recomputes every metric, interval diagnostic, Friedman test, Wilcoxon test, and Holm correction from that ledger.

The repository also contains executable PyTorch implementations of the compact LSTM, TCN, Transformer, and probabilistic LSTM. They can be independently retrained with:

```bash
python -m pip install -r requirements-neural.txt
python scripts/retrain_neural_models.py
```

Retrained outputs are written separately under `outputs/retrained_neural/` and compared with the immutable publication ledger. The archived ledger is not replaced automatically.

## Data structure

`data/indexed_monthly_data.csv` contains one row for each unique month from July 2020 to September 2024. June 2021 is normalized to 100 and occurs once. The main variables are:

- `cloud_cost_index`
- `vehicles_index`
- `total_messages_index`
- `cost_per_vehicle_index`
- `cost_per_billion_messages_index`

`data/implementation_series.csv` contains the 40-month boundary-to-endpoint series from June 2021 through September 2024. It is used for observed KPI reporting and counterfactual scenario plots.

All data are aggregated and indexed. No raw vehicle payloads or personally identifiable information are included.

## Analysis design

The workflow separates three forms of evidence:

1. **Directly observed outcomes:** total cost and demand-normalized unit costs.
2. **Model-dependent counterfactuals:** linear, quadratic, exponential, log-linear, and power-law cost-demand functions, with prediction intervals and demand scenarios.
3. **Quasi-experimental temporal evidence:** segmented regression of log cost per active vehicle with corrected Newey-West standard errors and implementation-lag sensitivity.

The common-fold forecasting benchmark uses six expanding-window, one-step-ahead origins. The first six months form the minimum training window, and months 7-12 are forecast sequentially without leakage.

## Reproducibility checks

`python reproduce.py` exits with a non-zero status if a core reference check fails. The checks include:

- quadratic coefficients and LOOCV RMSE;
- final counterfactual estimate;
- Friedman statistic and minimum Holm-adjusted p-value;
- primary interrupted time-series level and slope changes.

The detailed check table is written to `outputs/tables/reproducibility_checks.csv`.

## Public rounding and Table 3

The distributed index series is rounded for confidentiality. The pipeline therefore writes two diagnostic files:

- `table3_baseline_diagnostics_published.csv`: exact display values reported in the manuscript;
- `table3_diagnostics_recomputed_public_indices.csv`: a fresh recomputation from the released rounded index values.

The central counterfactual, sensitivity, benchmark, ITS, KPI, and uncertainty results are reproduced directly from the released values and prediction ledger.

## Repository structure

```text
.
├── data/                       # Indexed data and immutable prediction ledgers
├── docs/                       # Method and output documentation
├── scripts/                    # Optional neural retraining
├── src/finops_repro/           # Reusable analysis package
├── tests/                      # Numerical regression tests
├── .github/workflows/          # Automatic reproduction on GitHub Actions
├── reproduce.py                # Main entry point
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements*.txt
```

## Docker

```bash
docker build -t finops-ev-repro .
docker run --rm -v "$PWD/outputs:/app/outputs" finops-ev-repro
```

## Citation

Use the metadata in `CITATION.cff`. The article DOI should be added only after Elsevier assigns it.

## License

Code is released under the MIT License. The indexed case-study values are supplied solely to reproduce the associated scientific analysis; users remain responsible for observing any institutional or contractual restrictions applicable to derived data.
