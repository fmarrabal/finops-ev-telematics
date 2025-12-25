# Methodology

This document provides detailed technical documentation of the machine learning methodology used in this study.

## 1. Research Context

### 1.1 Problem Statement
Electric vehicle (EV) telematics platforms generate massive volumes of real-time data from connected vehicles, requiring substantial cloud infrastructure. As fleets scale from thousands to millions of vehicles, cloud costs can grow exponentially without proper governance. This study addresses the challenge of predicting and optimizing cloud costs in EV telematics platforms using machine learning methods within a FinOps framework.

### 1.2 Research Questions
1. What is the relationship between fleet size and cloud infrastructure costs?
2. Which ML forecasting methods best predict cloud costs in EV telematics?
3. How effective are FinOps interventions in reducing cloud costs?
4. How can counterfactual analysis quantify optimization impact?

## 2. FinOps Framework

### 2.1 Inform Phase
- Cost visibility through Azure Cost Management APIs
- Resource tagging and allocation
- Benchmarking against industry standards

### 2.2 Optimize Phase
- Rate optimization (reserved instances, savings plans)
- Usage optimization (rightsizing, autoscaling)
- Resource lifecycle management

### 2.3 Operate Phase
- Continuous monitoring with ML-based anomaly detection
- Governance policies and spending limits
- Weekly cost review cadences

## 3. Machine Learning Models

### 3.1 Polynomial Regression (Demand-Controlled Baseline)

**Rationale**: Cloud costs in telematics platforms are fundamentally driven by the number of connected vehicles generating data. A polynomial model captures this demand-cost relationship.

**Mathematical Formulation**:
```
C(V) = α + βV + γV²
```

Where:
- C = Cloud cost (indexed)
- V = Vehicle count (indexed)
- α = Fixed cost component
- β = Linear cost coefficient
- γ = Quadratic cost coefficient (captures economies/diseconomies of scale)

**Implementation**:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly_features.fit_transform(vehicles.reshape(-1, 1))

model = LinearRegression(fit_intercept=False)
model.fit(X_poly, costs)
```

**Fitted Equation**: `C(V) = 39.64 - 0.334V + 0.0094V²`

### 3.2 ARIMA (Autoregressive Integrated Moving Average)

**Rationale**: Time series modeling captures temporal autocorrelation in monthly cost data.

**Model Selection**: ARIMA(1,1,0) selected based on:
- AIC/BIC criteria
- Stationarity testing (ADF test)
- Residual diagnostics

**Implementation**:
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(1, 1, 0))
model_fit = model.fit()
forecast = model_fit.get_forecast(steps=n_forecast)
```

### 3.3 Facebook Prophet

**Rationale**: Automatic handling of trend components and robustness to missing data make Prophet suitable for business planning scenarios.

**Configuration**:
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=False,  # Only 12 months training data
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # Regularization
)
model.fit(prophet_df)
```

### 3.4 LSTM Neural Network

**Rationale**: Deep learning captures non-linear temporal patterns that statistical models may miss.

**Architecture**:
```
Input (lookback=3) → LSTM(50) → Dropout(0.1) → LSTM(30) → Dropout(0.1) → Dense(20) → Dense(1)
```

**Key Design Decisions**:
1. **Normalization**: MinMaxScaler(0.1, 0.9) to avoid sigmoid saturation
2. **Trend-guided extrapolation**: Blend LSTM predictions with linear trend for long-horizon stability
3. **Soft bounds**: Prevent explosive predictions during recursive forecasting

**Implementation**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.1),
    LSTM(30, activation='tanh'),
    Dropout(0.1),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
```

## 4. Evaluation Metrics

### 4.1 Metrics Computed

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | 1 - SS_res/SS_tot | Variance explained (0-1) |
| RMSE | √(Σ(y-ŷ)²/n) | Average error magnitude |
| MAE | Σ\|y-ŷ\|/n | Average absolute error |
| MAPE | 100×Σ\|(y-ŷ)/y\|/n | Percentage error |
| sMAPE | 100×Σ2\|y-ŷ\|/(y+ŷ)/n | Symmetric percentage error |

### 4.2 Cross-Validation

Leave-one-out cross-validation (LOO-CV) used for polynomial model to assess generalization:

```python
for i in range(n):
    # Train on all except observation i
    # Predict observation i
    # Record squared error
cv_rmse = sqrt(mean(squared_errors))
```

## 5. Counterfactual Analysis

### 5.1 Methodology

The counterfactual baseline represents the cost trajectory that would have occurred without FinOps interventions:

1. Fit polynomial model to pre-FinOps data (12 months)
2. Project vehicle growth into post-FinOps period
3. Apply cost-vehicle relationship to projected vehicles
4. Compare counterfactual costs with actual costs

### 5.2 Savings Attribution

```
Monthly Savings = Counterfactual Cost - Actual Cost
Cumulative Savings = Σ Monthly Savings
Efficiency Factor = Initial Unit Cost / Final Unit Cost
```

### 5.3 Uncertainty Quantification

95% prediction intervals computed using:
```
PI = ŷ ± 1.96 × σ_residual × distance_factor
```

Where `distance_factor` increases with extrapolation distance from training data.

## 6. Reproducibility

### 6.1 Random Seeds
```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

### 6.2 Software Versions
- Python 3.8+
- TensorFlow 2.6+
- Prophet 1.0+
- statsmodels 0.12+
- scikit-learn 0.24+

## 7. Limitations

1. **Training Data**: 12-month window limits LSTM learning capacity
2. **Single Platform**: Results from one OEM on Azure may not generalize
3. **Counterfactual Assumptions**: Assumes pre-FinOps relationship would persist
4. **LSTM Variability**: Neural network results may vary slightly between runs

## References

1. FinOps Foundation. (2024). FinOps Framework. https://www.finops.org/framework/
2. Box, G. E., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.
3. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. The American Statistician.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
