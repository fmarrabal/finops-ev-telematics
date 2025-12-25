"""
FinOps Cloud Cost Forecasting for EV Telematics Platforms
=========================================================

A machine learning framework for cloud cost forecasting and optimization
in electric vehicle telematics platforms.

Models:
    - PolynomialForecast: Demand-controlled polynomial regression
    - ARIMAForecast: ARIMA time series forecasting
    - ProphetForecast: Facebook Prophet forecasting
    - LSTMForecast: LSTM neural network forecasting

Utilities:
    - compute_metrics: Calculate R², RMSE, MAE, MAPE, sMAPE
    - plot_figure3_column: Generate publication-quality figures

Example usage:
    from finops_forecasting import run_complete_analysis, plot_figure3_column
    
    results = run_complete_analysis()
    fig = plot_figure3_column(results, save_path='figure3.png')
"""

# Import models
from .models import (
    PolynomialForecast,
    ARIMAForecast,
    ProphetForecast,
    LSTMForecast,
)

# Import utilities
from .utils import (
    compute_metrics,
    cross_validate_loo,
    plot_figure3_column,
    setup_publication_style,
    COLORS,
)

# Import main functions
from .finops_forecasting import (
    load_finops_data,
    run_complete_analysis,
    main,
)

__version__ = "1.0.0"
__author__ = "Víctor Valdivieso, Francisco Manuel Arrabal-Campos"
__email__ = "your-email@university.edu"

__all__ = [
    # Models
    'PolynomialForecast',
    'ARIMAForecast',
    'ProphetForecast',
    'LSTMForecast',
    # Utilities
    'compute_metrics',
    'cross_validate_loo',
    'plot_figure3_column',
    'setup_publication_style',
    'COLORS',
    # Main functions
    'load_finops_data',
    'run_complete_analysis',
    'main',
]
