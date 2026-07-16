# Reproduced methods

## 1. Monthly data and boundary handling

The calibration period is July 2020-June 2021. The implementation series also begins in June 2021. Full-series analyses count that shared boundary once, producing 51 unique observations. The implementation KPI series retains June 2021 as month 0 and therefore has 40 observations.

## 2. Demand-controlled counterfactual

The primary model is

```text
C(V) = alpha + beta V + gamma V^2.
```

It is estimated by ordinary least squares on the 12 calibration observations. Leave-one-out cross-validation selects the quadratic form among the low-parameter candidates. Observation-level prediction intervals use the standard OLS predictive variance.

Five functional forms are compared on the same data:

- linear;
- quadratic;
- exponential;
- log-linear;
- power law.

The final gap is `(counterfactual - observed) / counterfactual` at `V = 440.14`.

## 3. Common-fold forecasting benchmark

All ranked models are evaluated on the same six one-step-ahead origins. The earliest fold trains on months 1-6 and predicts month 7; the last trains on months 1-11 and predicts month 12.

ARIMA uses `ARIMA(1,1,0)` with a linear drift term for the common-fold comparison. The demand model uses the held-out contemporaneous vehicle index. Neural rows correspond to medians over five seeds in the publication ledger.

The global test is Friedman's test on per-origin absolute errors. Pairwise comparisons use one-sided exact Wilcoxon signed-rank tests of whether ARIMA absolute errors are lower, followed by Holm correction over six comparisons.

## 4. Interrupted time series

For monthly cost per active vehicle `U_t = 100 C_t / V_t`, the primary model is

```text
log(U_t) = beta0 + beta1 t + beta2 I_t + beta3 (t - T0) I_t + error_t.
```

The primary effect start is July 2021. June, August, and September starts are sensitivity checks. Covariance estimates use heteroskedasticity- and autocorrelation-consistent Newey-West standard errors with three lags and finite-sample correction.

Percentage effects are `100 * (exp(beta) - 1)`.

## 5. Uncertainty

The quadratic counterfactual reports 50%, 80%, and 95% observation-level prediction intervals. Horizon summaries use observed vehicle indices at implementation months 6, 12, 24, and 39. Final low and high scenarios use 85% and 115% of the final observed vehicle index.
