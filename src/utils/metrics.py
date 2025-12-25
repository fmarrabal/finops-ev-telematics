#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics computation utilities for model evaluation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for forecasting models.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for identification
        
    Returns
    -------
    dict
        Dictionary containing R², RMSE, MAE, MAPE, and sMAPE metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    # Compute metrics
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


def cross_validate_loo(X, y, model_class, **model_params):
    """
    Perform Leave-One-Out cross-validation.
    
    Parameters
    ----------
    X : array-like
        Feature values
    y : array-like
        Target values
    model_class : class
        Model class with fit() and predict() methods
    **model_params : dict
        Parameters to pass to model constructor
        
    Returns
    -------
    float
        Cross-validated RMSE
    """
    X = np.array(X)
    y = np.array(y)
    errors = []
    
    for i in range(len(X)):
        # Leave one out
        X_train = np.delete(X, i)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]
        
        # Fit and predict
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        errors.append((y_test - y_pred) ** 2)
    
    return np.sqrt(np.mean(errors))
