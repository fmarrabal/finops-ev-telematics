#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinOps Cloud Cost Forecasting - FINAL VERSION
==============================================
Complete metrics calculation and Figure 3 in vertical column layout.

Author: Francisco Manuel Arrabal
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Statistical models
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Prophet
from prophet import Prophet

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# PLOTTING CONFIGURATION - Publication Quality
# =============================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_finops_data():
    """Load and preprocess the FinOps case study data."""
    
    pre_finops = pd.DataFrame({
        'month_index': np.arange(12),
        'date': pd.date_range(start='2020-07-01', periods=12, freq='MS'),
        'cloud_cost_index': [40.86, 42.02, 41.34, 43.06, 47.65, 52.70, 
                            59.87, 65.21, 72.12, 79.45, 88.34, 100.00],
        'vehicles_index': [35.43, 38.60, 42.31, 47.65, 53.94, 59.76, 
                          64.94, 71.34, 79.32, 85.19, 91.59, 100.00]
    })
    
    post_finops_values = {
        'total_messages_index': [100.0, 116.10, 120.81, 135.27, 159.11, 174.43, 
            188.60, 205.75, 202.69, 257.52, 270.20, 300.10, 321.49, 339.48, 
            347.55, 403.91, 478.33, 499.74, 482.72, 486.99, 488.23, 610.16, 
            619.27, 688.41, 713.32, 725.69, 716.88, 738.11, 787.26, 794.52, 
            816.17, 811.69, 814.61, 914.36, 926.12, 964.46, 952.31, 990.89, 
            937.26, 1006.85],
        'vehicles_index': [100.0, 105.54, 110.33, 116.06, 122.71, 128.95, 
            135.86, 141.05, 147.52, 154.53, 159.09, 164.47, 171.23, 177.05, 
            182.48, 190.72, 192.75, 203.61, 209.46, 219.80, 230.37, 242.93, 
            251.88, 263.69, 275.07, 286.50, 293.89, 306.03, 317.74, 329.43, 
            339.07, 349.26, 361.04, 373.20, 383.92, 396.53, 408.43, 421.20, 
            427.32, 440.14],
        'cloud_cost_index': [100.0, 78.45, 64.82, 55.77, 57.27, 53.34, 
            55.02, 58.80, 58.04, 60.19, 43.59, 45.00, 48.35, 46.50, 43.54, 
            43.77, 39.42, 37.95, 37.86, 37.35, 35.33, 39.00, 42.24, 44.16, 
            44.68, 44.60, 43.15, 42.03, 43.59, 43.00, 45.24, 45.54, 42.86, 
            44.25, 45.02, 47.72, 39.61, 43.18, 44.40, 45.08],
        'cost_per_billion_msg': [100.0, 67.57, 53.65, 41.23, 36.00, 30.58, 
            29.17, 28.58, 28.64, 23.37, 16.13, 15.00, 15.04, 14.46, 12.53, 
            10.84, 8.24, 7.59, 7.84, 7.67, 7.24, 6.39, 6.82, 6.41, 6.26, 
            6.15, 6.02, 5.69, 5.54, 5.41, 5.54, 5.61, 5.26, 4.84, 4.86, 
            4.95, 4.16, 4.36, 4.74, 4.48]
    }
    
    n_months = len(post_finops_values['total_messages_index'])
    post_finops = pd.DataFrame({
        'month_index': np.arange(n_months),
        'date': pd.date_range(start='2021-06-01', periods=n_months, freq='MS'),
        **post_finops_values
    })
    
    return pre_finops, post_finops


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true, y_pred, model_name="Model"):
    """Compute comprehensive evaluation metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'R²': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE (%)': round(mape, 4),
        'sMAPE (%)': round(smape, 4)
    }


# =============================================================================
# POLYNOMIAL REGRESSION
# =============================================================================

class PolynomialForecast:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None
        self.poly_features = None
        self.residual_std = None
        self.X_train = None
        
    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).flatten()
        self.X_train = X
        
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=True)
        X_poly = self.poly_features.fit_transform(X)
        
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X_poly, y)
        
        y_pred = self.model.predict(X_poly)
        self.residual_std = np.std(y - y_pred)
        return self
    
    def predict(self, X, return_interval=False, z=1.96):
        X = np.array(X).reshape(-1, 1)
        X_poly = self.poly_features.transform(X)
        y_pred = self.model.predict(X_poly)
        
        if return_interval:
            X_mean = np.mean(self.X_train)
            distance_factor = 1 + 0.1 * np.abs(X.flatten() - X_mean) / (np.std(self.X_train) + 1e-10)
            interval = z * self.residual_std * distance_factor
            return y_pred, y_pred - interval, y_pred + interval
        return y_pred


# =============================================================================
# COMPLETE ANALYSIS WITH ALL METRICS
# =============================================================================

def run_complete_analysis():
    """Run complete analysis with all metrics calculated properly."""
    
    print("="*70)
    print("FinOps Cloud Cost Forecasting - COMPLETE ANALYSIS")
    print("="*70)
    
    # Load data
    pre_finops, post_finops = load_finops_data()
    
    y_train = pre_finops['cloud_cost_index'].values
    X_train = pre_finops['vehicles_index'].values
    dates_train = pre_finops['date'].values
    n_forecast = len(post_finops)
    
    metrics_list = []
    forecasts = {}
    
    # =========================================================================
    # 1. POLYNOMIAL REGRESSION
    # =========================================================================
    print("\n[1/4] Polynomial Regression (degree=2)...")
    
    poly_features = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
    
    poly_model = LinearRegression(fit_intercept=False)
    poly_model.fit(X_poly, y_train)
    
    y_pred_poly = poly_model.predict(X_poly)
    metrics_poly = compute_metrics(y_train, y_pred_poly, "Polynomial (degree 2)")
    metrics_list.append(metrics_poly)
    
    coefs = poly_model.coef_
    print(f"  Equation: C(V) = {coefs[0]:.2f} + {coefs[1]:.3f}V + {coefs[2]:.4f}V²")
    print(f"  R² = {metrics_poly['R²']:.4f}, RMSE = {metrics_poly['RMSE']:.3f}, MAE = {metrics_poly['MAE']:.3f}")
    print(f"  MAPE = {metrics_poly['MAPE (%)']:.2f}%, sMAPE = {metrics_poly['sMAPE (%)']:.2f}%")
    
    # Forecast
    poly_cost = PolynomialForecast(degree=2)
    poly_cost.fit(X_train, y_train)
    
    poly_veh = PolynomialForecast(degree=2)
    poly_veh.fit(pre_finops['month_index'].values, X_train)
    
    future_months = np.arange(12, 12 + n_forecast)
    vehicles_projected = poly_veh.predict(future_months)
    cost_cf, cost_cf_lower, cost_cf_upper = poly_cost.predict(vehicles_projected, return_interval=True)
    
    forecasts['Polynomial'] = {'pred': cost_cf, 'lower': cost_cf_lower, 'upper': cost_cf_upper}
    
    # =========================================================================
    # 2. ARIMA
    # =========================================================================
    print("\n[2/4] ARIMA (1,1,0)...")
    
    arima_model = ARIMA(y_train, order=(1, 1, 0))
    arima_fit = arima_model.fit()
    
    y_fitted_arima = arima_fit.fittedvalues
    if hasattr(y_fitted_arima, 'values'):
        y_fitted_arima = y_fitted_arima.values
    
    metrics_arima = compute_metrics(y_train[1:], y_fitted_arima[1:], "ARIMA (1,1,0)")
    metrics_list.append(metrics_arima)
    
    print(f"  R² = {metrics_arima['R²']:.4f}, RMSE = {metrics_arima['RMSE']:.3f}, MAE = {metrics_arima['MAE']:.3f}")
    print(f"  MAPE = {metrics_arima['MAPE (%)']:.2f}%, sMAPE = {metrics_arima['sMAPE (%)']:.2f}%")
    
    # Forecast
    forecast = arima_fit.get_forecast(steps=n_forecast)
    arima_pred = forecast.predicted_mean.values if hasattr(forecast.predicted_mean, 'values') else np.array(forecast.predicted_mean)
    conf_int = forecast.conf_int(alpha=0.05)
    if hasattr(conf_int, 'values'):
        conf_arr = conf_int.values
    else:
        conf_arr = np.array(conf_int)
    
    forecasts['ARIMA'] = {'pred': arima_pred, 'lower': conf_arr[:, 0], 'upper': conf_arr[:, 1]}
    
    # =========================================================================
    # 3. FACEBOOK PROPHET
    # =========================================================================
    print("\n[3/4] Facebook Prophet...")
    
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(dates_train),
        'y': y_train
    })
    
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(prophet_df)
    
    prophet_pred_train = prophet_model.predict(prophet_df)
    y_pred_prophet = prophet_pred_train['yhat'].values
    
    metrics_prophet = compute_metrics(y_train, y_pred_prophet, "Facebook Prophet")
    metrics_list.append(metrics_prophet)
    
    print(f"  R² = {metrics_prophet['R²']:.4f}, RMSE = {metrics_prophet['RMSE']:.3f}, MAE = {metrics_prophet['MAE']:.3f}")
    print(f"  MAPE = {metrics_prophet['MAPE (%)']:.2f}%, sMAPE = {metrics_prophet['sMAPE (%)']:.2f}%")
    
    # Forecast
    future_dates = pd.date_range(start='2021-06-01', periods=n_forecast, freq='MS')
    future_df = pd.DataFrame({'ds': future_dates})
    prophet_forecast = prophet_model.predict(future_df)
    
    forecasts['Prophet'] = {
        'pred': prophet_forecast['yhat'].values,
        'lower': prophet_forecast['yhat_lower'].values,
        'upper': prophet_forecast['yhat_upper'].values
    }
    
    # =========================================================================
    # 4. LSTM NEURAL NETWORK (with proper in-sample metrics)
    # =========================================================================
    print("\n[4/4] LSTM Neural Network...")
    
    lookback = 3
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    y_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_lstm, y_lstm = [], []
    for i in range(len(y_scaled) - lookback):
        X_lstm.append(y_scaled[i:i+lookback])
        y_lstm.append(y_scaled[i+lookback])
    X_lstm = np.array(X_lstm).reshape(-1, lookback, 1)
    y_lstm = np.array(y_lstm)
    
    # Build model
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        LSTM(30, activation='tanh'),
        Dropout(0.1),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
    model.fit(X_lstm, y_lstm, epochs=300, batch_size=2, callbacks=[early_stop], verbose=0)
    
    # In-sample predictions for metrics
    y_pred_lstm_scaled = model.predict(X_lstm, verbose=0).flatten()
    y_pred_lstm_insample = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
    y_actual_lstm = y_train[lookback:]
    
    metrics_lstm = compute_metrics(y_actual_lstm, y_pred_lstm_insample, "LSTM Neural Network")
    metrics_list.append(metrics_lstm)
    
    print(f"  R² = {metrics_lstm['R²']:.4f}, RMSE = {metrics_lstm['RMSE']:.3f}, MAE = {metrics_lstm['MAE']:.3f}")
    print(f"  MAPE = {metrics_lstm['MAPE (%)']:.2f}%, sMAPE = {metrics_lstm['sMAPE (%)']:.2f}%")
    print(f"  (Evaluated on {len(y_actual_lstm)} points, n - lookback)")
    
    # Forecast with trend guidance
    predictions = []
    current_seq = y_scaled[-lookback:]
    trend_slope = np.polyfit(np.arange(len(y_train)), y_train, 1)[0]
    
    for step in range(n_forecast):
        X_pred = current_seq.reshape(1, lookback, 1)
        next_val_scaled = model.predict(X_pred, verbose=0)[0, 0]
        next_val = scaler.inverse_transform([[next_val_scaled]])[0, 0]
        
        # Blend with trend
        expected_val = y_train[-1] + trend_slope * (step + 1)
        trend_weight = min(0.3 + step * 0.015, 0.6)
        next_val = (1 - trend_weight) * next_val + trend_weight * expected_val
        
        # Bounds
        max_allowed = y_train.max() + (y_train.max() - y_train.min()) * (step + 1) * 0.12
        next_val = np.clip(next_val, y_train.min() * 0.5, max_allowed)
        
        predictions.append(next_val)
        
        next_norm = scaler.transform([[next_val]])[0, 0]
        next_norm = np.clip(next_norm, 0.1, 0.9)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = next_norm
    
    lstm_pred = np.array(predictions)
    std_est = np.std(y_train) * 0.3
    horizon_factor = np.sqrt(np.arange(1, n_forecast + 1)) / 3
    interval = 1.96 * std_est * horizon_factor
    
    forecasts['LSTM'] = {
        'pred': lstm_pred,
        'lower': lstm_pred - interval,
        'upper': lstm_pred + interval
    }
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL MODEL METRICS (In-Sample Performance)")
    print("="*70)
    
    results_df = pd.DataFrame(metrics_list)
    
    print("\n{:<22} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "Model", "R²", "RMSE", "MAE", "MAPE (%)", "sMAPE (%)"))
    print("-"*70)
    
    for r in metrics_list:
        print("{:<22} {:>8.4f} {:>8.3f} {:>8.3f} {:>10.2f} {:>10.2f}".format(
            r['Model'], r['R²'], r['RMSE'], r['MAE'], r['MAPE (%)'], r['sMAPE (%)']))
    
    # Cross-validation for polynomial
    print("\n" + "="*70)
    print("CROSS-VALIDATION (Leave-One-Out)")
    print("="*70)
    
    loo_errors = []
    for i in range(len(X_train)):
        X_cv = np.delete(X_train, i)
        y_cv = np.delete(y_train, i)
        
        pf = PolynomialFeatures(degree=2, include_bias=True)
        X_poly_cv = pf.fit_transform(X_cv.reshape(-1, 1))
        model_cv = LinearRegression(fit_intercept=False)
        model_cv.fit(X_poly_cv, y_cv)
        
        X_test = pf.transform(X_train[i].reshape(-1, 1))
        y_pred = model_cv.predict(X_test)[0]
        loo_errors.append((y_train[i] - y_pred)**2)
    
    cv_rmse = np.sqrt(np.mean(loo_errors))
    print(f"  Polynomial LOO-CV RMSE: {cv_rmse:.2f}")
    
    return {
        'pre_finops': pre_finops,
        'post_finops': post_finops,
        'forecasts': forecasts,
        'metrics': results_df,
        'counterfactual': {
            'vehicles_projected': vehicles_projected,
            'cost_counterfactual': cost_cf,
            'cost_cf_lower': cost_cf_lower,
            'cost_cf_upper': cost_cf_upper,
            'months': future_months
        }
    }


# =============================================================================
# FIGURE 3 - VERTICAL COLUMN LAYOUT (3 panels)
# =============================================================================

def plot_figure3_column(results, save_path=None):
    """
    Generate Figure 3 with 3 panels in VERTICAL COLUMN layout.
    
    Panel (a): Forecasting models comparison
    Panel (b): Counterfactual analysis  
    Panel (c): Efficiency evolution
    """
    
    pre_finops = results['pre_finops']
    post_finops = results['post_finops']
    forecasts = results['forecasts']
    counterfactual = results['counterfactual']
    
    # Create figure with 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    
    # Color scheme
    colors = {
        'observed': '#2E86AB',      # Blue
        'polynomial': '#A23B72',    # Magenta
        'arima': '#F18F01',         # Orange  
        'prophet': '#C73E1D',       # Red
        'lstm': '#3B1F2B',          # Dark purple
        'actual': '#28A745',        # Green
        'counterfactual': '#DC3545' # Red
    }
    
    n_forecast = len(post_finops)
    future_months = np.arange(12, 12 + n_forecast)
    
    # =========================================================================
    # Panel (a): Forecasting Models Comparison
    # =========================================================================
    ax1 = axes[0]
    
    # Observed data (pre-FinOps)
    ax1.scatter(pre_finops['month_index'], pre_finops['cloud_cost_index'], 
                s=80, c=colors['observed'], label='Observed (Pre-FinOps)', 
                zorder=5, edgecolors='white', linewidth=1)
    
    # Plot each model forecast
    max_reasonable = 400
    model_colors = {
        'Polynomial': colors['polynomial'],
        'ARIMA': colors['arima'],
        'Prophet': colors['prophet'],
        'LSTM': colors['lstm']
    }
    
    for name, data in forecasts.items():
        if data['pred'] is not None:
            pred = np.array(data['pred'])
            if np.max(pred) < max_reasonable:
                ax1.plot(future_months, pred, '-', linewidth=2, 
                         color=model_colors[name], label=f'{name}', alpha=0.9)
                
                if data['lower'] is not None:
                    lower = np.clip(data['lower'], 0, max_reasonable)
                    upper = np.clip(data['upper'], 0, max_reasonable)
                    ax1.fill_between(future_months, lower, upper, 
                                     alpha=0.15, color=model_colors[name])
    
    ax1.axvline(x=11.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(10, 350, 'Forecast\nStart', ha='right', va='top', fontsize=9, color='gray')
    
    ax1.set_xlabel('Month Index (from Jul 2020)')
    ax1.set_ylabel('Cloud Cost (Indexed, Jun 2021 = 100)')
    ax1.set_title('(a) Forecasting Models Comparison', fontweight='bold', pad=10, loc='left')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(0, max_reasonable)
    ax1.set_xlim(-1, 55)
    
    # =========================================================================
    # Panel (b): Counterfactual Analysis (FinOps Impact)
    # =========================================================================
    ax2 = axes[1]
    
    # Pre-FinOps observed
    ax2.scatter(pre_finops['month_index'], pre_finops['cloud_cost_index'], 
                s=80, c=colors['observed'], label='Pre-FinOps Observed', 
                zorder=5, edgecolors='white', linewidth=1)
    
    # Counterfactual projection with 95% PI (orange shading)
    cf_months = counterfactual['months']
    cf_cost = counterfactual['cost_counterfactual']
    cf_lower = counterfactual['cost_cf_lower']
    cf_upper = counterfactual['cost_cf_upper']
    
    ax2.fill_between(cf_months, cf_lower, cf_upper, 
                     alpha=0.25, color='#FFA500', label='95% Prediction Interval')
    ax2.plot(cf_months, cf_cost, '--', color=colors['counterfactual'], 
             linewidth=2.5, label='Counterfactual (No FinOps)')
    
    # Actual post-FinOps cost
    actual_months = np.arange(12, 12 + len(post_finops))
    ax2.scatter(actual_months, post_finops['cloud_cost_index'], 
                s=50, c=colors['actual'], alpha=0.7, label='Actual (With FinOps)',
                edgecolors='white', linewidth=0.5, zorder=4)
    
    ax2.axvline(x=11.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Shade savings area
    actual_cost = post_finops['cloud_cost_index'].values
    ax2.fill_between(actual_months, actual_cost, cf_cost, 
                     alpha=0.3, color='green', label='FinOps Savings')
    
    ax2.set_xlabel('Month Index (from Jul 2020)')
    ax2.set_ylabel('Cloud Cost (Indexed)')
    ax2.set_title('(b) FinOps Impact: Counterfactual vs Actual', fontweight='bold', pad=10, loc='left')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax2.set_ylim(0, min(500, max(cf_upper) * 1.1))
    ax2.set_xlim(-1, 55)
    
    # =========================================================================
    # Panel (c): Cost Efficiency Evolution
    # =========================================================================
    ax3 = axes[2]
    
    months_post = post_finops['month_index'].values
    cost_per_billion = post_finops['cost_per_billion_msg'].values
    
    ax3.plot(months_post, cost_per_billion, '-', color=colors['actual'], 
             linewidth=2.5, marker='o', markersize=4, label='Cost per Billion Messages')
    
    ax3.axhline(y=cost_per_billion[0], color=colors['counterfactual'], 
                linestyle='--', alpha=0.7, linewidth=1.5, label='Initial Level (100)')
    
    ax3.fill_between(months_post, cost_per_billion, cost_per_billion[0], 
                     alpha=0.3, color='green')
    
    efficiency_factor = cost_per_billion[0] / cost_per_billion[-1]
    ax3.annotate(f'{efficiency_factor:.1f}× improvement', 
                 xy=(months_post[-1], cost_per_billion[-1]),
                 xytext=(months_post[-1]-12, cost_per_billion[-1]+20),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    ax3.set_xlabel('Month Index (Post-FinOps)')
    ax3.set_ylabel('Cost per Billion Messages (Indexed)')
    ax3.set_title('(c) Cloud Cost Efficiency Evolution', fontweight='bold', pad=10, loc='left')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0, 110)
    
    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    # Run complete analysis
    results = run_complete_analysis()
    
    # Generate Figure 3 in column layout
    print("\n" + "="*70)
    print("GENERATING FIGURE 3 (VERTICAL COLUMN LAYOUT)")
    print("="*70)
    
    fig = plot_figure3_column(results, save_path='Figure3_column.png')
    
    # Save metrics
    results['metrics'].to_csv('model_metrics_final.csv', index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - Figure3_column.png")
    print("  - model_metrics_final.csv")
