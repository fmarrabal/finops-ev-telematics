#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Facebook Prophet model for time series forecasting.
"""

import numpy as np
import pandas as pd
from prophet import Prophet


class ProphetForecast:
    """
    Facebook Prophet forecasting model.
    
    Prophet is designed for business time series with strong seasonal effects
    and several seasons of historical data. It is robust to missing data,
    shifts in trend, and typically handles outliers well.
    
    Parameters
    ----------
    yearly_seasonality : bool, default=False
        Whether to include yearly seasonality
    weekly_seasonality : bool, default=False
        Whether to include weekly seasonality
    daily_seasonality : bool, default=False
        Whether to include daily seasonality
    changepoint_prior_scale : float, default=0.05
        Flexibility of trend changepoints (higher = more flexible)
        
    Attributes
    ----------
    model : Prophet
        Prophet model instance
    fitted_values : ndarray
        In-sample predictions
        
    Example
    -------
    >>> model = ProphetForecast()
    >>> model.fit(dates, values)
    >>> forecast = model.forecast(future_dates)
    """
    
    def __init__(self, yearly_seasonality=False, weekly_seasonality=False,
                 daily_seasonality=False, changepoint_prior_scale=0.05):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.fitted_values = None
        
    def fit(self, dates, y):
        """
        Fit the Prophet model.
        
        Parameters
        ----------
        dates : array-like
            Datetime values for the time series
        y : array-like
            Target values
            
        Returns
        -------
        self
            Fitted model instance
        """
        # Prepare data in Prophet format
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': np.array(y).flatten()
        })
        
        # Initialize and fit model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        self.model.fit(df)
        
        # Store fitted values
        fitted = self.model.predict(df)
        self.fitted_values = fitted['yhat'].values
        
        return self
    
    def forecast(self, future_dates):
        """
        Generate forecasts for future dates.
        
        Parameters
        ----------
        future_dates : array-like
            Datetime values for forecast period
            
        Returns
        -------
        dict
            Dictionary with 'pred', 'lower', and 'upper' keys
        """
        future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
        forecast = self.model.predict(future_df)
        
        return {
            'pred': forecast['yhat'].values,
            'lower': forecast['yhat_lower'].values,
            'upper': forecast['yhat_upper'].values
        }
    
    def get_fitted_values(self):
        """
        Return in-sample fitted values.
        
        Returns
        -------
        ndarray
            Fitted values for training data
        """
        return self.fitted_values
    
    def get_components(self, dates):
        """
        Get trend and seasonality components.
        
        Parameters
        ----------
        dates : array-like
            Datetime values
            
        Returns
        -------
        DataFrame
            DataFrame with trend and seasonal components
        """
        df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        return self.model.predict(df)[['ds', 'trend', 'yhat']]
