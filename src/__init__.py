"""
FinOps Cloud Cost Forecasting for EV Telematics Platforms
=========================================================

A machine learning framework for cloud cost forecasting and optimization
in electric vehicle telematics platforms.

Models included:
- Polynomial Regression (demand-controlled baseline)
- ARIMA (time series forecasting)
- Facebook Prophet (trend/seasonality handling)
- LSTM Neural Network (deep learning)

Example usage:
    from finops_forecasting import run_complete_analysis, plot_figure3_column
    
    results = run_complete_analysis()
    fig = plot_figure3_column(results, save_path='figure3.png')
"""

from .finops_forecasting import (
    load_finops_data,
    compute_metrics,
    PolynomialForecast,
    run_complete_analysis,
    plot_figure3_column,
)

__version__ = "1.0.0"
__author__ = "Víctor Valdivieso, Francisco Manuel Arrabal-Campos"
__email__ = "your-email@university.edu"

__all__ = [
    "load_finops_data",
    "compute_metrics",
    "PolynomialForecast",
    "run_complete_analysis",
    "plot_figure3_column",
]
