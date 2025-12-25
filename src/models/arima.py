#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA model for time series forecasting.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAForecast:
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecasting model.
    
    Parameters
    ----------
    order : tuple, default=(1, 1, 0)
        The (p, d, q) order of the ARIMA model
        - p: AR order (autoregressive)
        - d: Differencing order
        - q: MA order (moving average)
        
    Attributes
    ----------
    model : ARIMA
        Statsmodels ARIMA model
    model_fit : ARIMAResults
        Fitted model results
    fitted_values : ndarray
        In-sample fitted values
        
    Example
    -------
    >>> model = ARIMAForecast(order=(1, 1, 0))
    >>> model.fit(cost_series)
    >>> forecast, conf_int = model.forecast(steps=12)
    """
    
    def __init__(self, order=(1, 1, 0)):
        self.order = order
        self.model = None
        self.model_fit = None
        self.fitted_values = None
        
    def fit(self, y):
        """
        Fit the ARIMA model to time series data.
        
        Parameters
        ----------
        y : array-like
            Time series values
            
        Returns
        -------
        self
            Fitted model instance
        """
        y = np.array(y).flatten()
        
        self.model = ARIMA(y, order=self.order)
        self.model_fit = self.model.fit()
        
        # Store fitted values
        self.fitted_values = self.model_fit.fittedvalues
        if hasattr(self.fitted_values, 'values'):
            self.fitted_values = self.fitted_values.values
            
        return self
    
    def forecast(self, steps, alpha=0.05):
        """
        Generate forecasts with confidence intervals.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        alpha : float, default=0.05
            Significance level for confidence intervals (0.05 = 95% CI)
            
        Returns
        -------
        tuple
            (predictions, confidence_intervals)
            - predictions: ndarray of point forecasts
            - confidence_intervals: ndarray of shape (steps, 2) with lower and upper bounds
        """
        forecast_result = self.model_fit.get_forecast(steps=steps)
        
        predictions = forecast_result.predicted_mean
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)
            
        conf_int = forecast_result.conf_int(alpha=alpha)
        if hasattr(conf_int, 'values'):
            conf_int = conf_int.values
        else:
            conf_int = np.array(conf_int)
            
        return predictions, conf_int
    
    def get_fitted_values(self):
        """
        Return in-sample fitted values.
        
        Returns
        -------
        ndarray
            Fitted values for training data
        """
        return self.fitted_values
    
    def summary(self):
        """
        Return model summary.
        
        Returns
        -------
        str
            Model summary from statsmodels
        """
        return self.model_fit.summary()
    
    def get_aic(self):
        """Return AIC (Akaike Information Criterion)."""
        return self.model_fit.aic
    
    def get_bic(self):
        """Return BIC (Bayesian Information Criterion)."""
        return self.model_fit.bic
