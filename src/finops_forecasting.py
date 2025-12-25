#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinOps Cloud Cost Forecasting - Main Analysis Script
=====================================================

This script runs the complete FinOps cloud cost forecasting analysis,
comparing multiple machine learning methods and generating publication-quality
visualizations.

Authors: Víctor Valdivieso, Francisco Manuel Arrabal-Campos
Date: December 2025

Usage:
    python finops_forecasting.py

Output:
    - Figure3_column.png: Publication figure with 3 panels
    - model_metrics_final.csv: Performance metrics for all models
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

# Import models
from models import PolynomialForecast, ARIMAForecast, ProphetForecast, LSTMForecast

# Import utilities
from utils import compute_metrics, plot_figure3_column

# For standalone polynomial fitting
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# =============================================================================
# DATA LOADING
# =============================================================================

def load_finops_data():
    """
    Load and preprocess the FinOps case study data.
    
    Returns
    -------
    tuple
        (pre_finops, post_finops) DataFrames
        
    Notes
    -----
    Pre-FinOps: July 2020 - June 2021 (12 months)
    Post-FinOps: June 2021 - September 2024 (40 months)
    All values indexed with June 2021 = 100
    """
    
    # Pre-FinOps data (Jul 2020 - Jun 2021)
    pre_finops = pd.DataFrame({
        'month_index': np.arange(12),
        'date': pd.date_range(start='2020-07-01', periods=12, freq='MS'),
        'cloud_cost_index': [40.86, 42.02, 41.34, 43.06, 47.65, 52.70, 
                            59.87, 65.21, 72.12, 79.45, 88.34, 100.00],
        'vehicles_index': [35.43, 38.60, 42.31, 47.65, 53.94, 59.76, 
                          64.94, 71.34, 79.32, 85.19, 91.59, 100.00]
    })
    
    # Post-FinOps data (Jun 2021 - Sep 2024)
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
# COMPLETE ANALYSIS
# =============================================================================

def run_complete_analysis():
    """
    Run complete analysis with all forecasting models.
    
    Returns
    -------
    dict
        Results dictionary containing:
        - pre_finops: Pre-FinOps DataFrame
        - post_finops: Post-FinOps DataFrame
        - forecasts: Dictionary of model forecasts
        - metrics: DataFrame of model performance metrics
        - counterfactual: Dictionary of counterfactual analysis results
    """
    
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
    
    # Fit using sklearn for metrics
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
    
    # Forecast using PolynomialForecast class
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
    
    arima_model = ARIMAForecast(order=(1, 1, 0))
    arima_model.fit(y_train)
    
    y_fitted_arima = arima_model.get_fitted_values()
    metrics_arima = compute_metrics(y_train[1:], y_fitted_arima[1:], "ARIMA (1,1,0)")
    metrics_list.append(metrics_arima)
    
    print(f"  R² = {metrics_arima['R²']:.4f}, RMSE = {metrics_arima['RMSE']:.3f}, MAE = {metrics_arima['MAE']:.3f}")
    print(f"  MAPE = {metrics_arima['MAPE (%)']:.2f}%, sMAPE = {metrics_arima['sMAPE (%)']:.2f}%")
    
    # Forecast
    arima_pred, arima_conf = arima_model.forecast(steps=n_forecast)
    forecasts['ARIMA'] = {'pred': arima_pred, 'lower': arima_conf[:, 0], 'upper': arima_conf[:, 1]}
    
    # =========================================================================
    # 3. FACEBOOK PROPHET
    # =========================================================================
    print("\n[3/4] Facebook Prophet...")
    
    prophet_model = ProphetForecast(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(dates_train, y_train)
    
    y_pred_prophet = prophet_model.get_fitted_values()
    metrics_prophet = compute_metrics(y_train, y_pred_prophet, "Facebook Prophet")
    metrics_list.append(metrics_prophet)
    
    print(f"  R² = {metrics_prophet['R²']:.4f}, RMSE = {metrics_prophet['RMSE']:.3f}, MAE = {metrics_prophet['MAE']:.3f}")
    print(f"  MAPE = {metrics_prophet['MAPE (%)']:.2f}%, sMAPE = {metrics_prophet['sMAPE (%)']:.2f}%")
    
    # Forecast
    future_dates = pd.date_range(start='2021-06-01', periods=n_forecast, freq='MS')
    prophet_forecast = prophet_model.forecast(future_dates)
    forecasts['Prophet'] = prophet_forecast
    
    # =========================================================================
    # 4. LSTM NEURAL NETWORK
    # =========================================================================
    print("\n[4/4] LSTM Neural Network...")
    
    lstm_model = LSTMForecast(
        lookback=3,
        lstm_units=[50, 30],
        dropout_rate=0.1,
        learning_rate=0.005,
        epochs=300,
        batch_size=2,
        patience=30
    )
    lstm_model.fit(y_train, verbose=0)
    
    y_pred_lstm = lstm_model.get_fitted_values()
    y_actual_lstm = y_train[lstm_model.lookback:]
    
    metrics_lstm = compute_metrics(y_actual_lstm, y_pred_lstm, "LSTM Neural Network")
    metrics_list.append(metrics_lstm)
    
    print(f"  R² = {metrics_lstm['R²']:.4f}, RMSE = {metrics_lstm['RMSE']:.3f}, MAE = {metrics_lstm['MAE']:.3f}")
    print(f"  MAPE = {metrics_lstm['MAPE (%)']:.2f}%, sMAPE = {metrics_lstm['sMAPE (%)']:.2f}%")
    print(f"  (Evaluated on {len(y_actual_lstm)} points, n - lookback)")
    
    # Forecast
    lstm_forecast = lstm_model.forecast(steps=n_forecast)
    forecasts['LSTM'] = lstm_forecast
    
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
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the analysis."""
    
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
    
    return results


if __name__ == "__main__":
    main()
