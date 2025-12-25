#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial Regression model for demand-controlled cost forecasting.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PolynomialForecast:
    """
    Polynomial regression model with prediction intervals.
    
    This model captures the relationship between a demand variable (e.g., vehicle count)
    and cloud costs, enabling demand-controlled forecasting.
    
    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial features
        
    Attributes
    ----------
    model : LinearRegression
        Fitted linear regression model
    poly_features : PolynomialFeatures
        Polynomial feature transformer
    residual_std : float
        Standard deviation of residuals for prediction intervals
    X_train : ndarray
        Training feature values (for distance-based interval widening)
        
    Example
    -------
    >>> model = PolynomialForecast(degree=2)
    >>> model.fit(vehicles, costs)
    >>> predictions = model.predict(future_vehicles)
    >>> pred, lower, upper = model.predict(future_vehicles, return_interval=True)
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None
        self.poly_features = None
        self.residual_std = None
        self.X_train = None
        
    def fit(self, X, y):
        """
        Fit the polynomial regression model.
        
        Parameters
        ----------
        X : array-like
            Feature values (e.g., vehicle count)
        y : array-like
            Target values (e.g., cloud cost)
            
        Returns
        -------
        self
            Fitted model instance
        """
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).flatten()
        self.X_train = X
        
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=True)
        X_poly = self.poly_features.fit_transform(X)
        
        # Fit linear regression
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X_poly, y)
        
        # Compute residual standard deviation for prediction intervals
        y_pred = self.model.predict(X_poly)
        self.residual_std = np.std(y - y_pred)
        
        return self
    
    def predict(self, X, return_interval=False, z=1.96):
        """
        Generate predictions with optional prediction intervals.
        
        Parameters
        ----------
        X : array-like
            Feature values for prediction
        return_interval : bool, default=False
            Whether to return prediction intervals
        z : float, default=1.96
            Z-score for confidence level (1.96 for 95% CI)
            
        Returns
        -------
        ndarray or tuple
            Predictions, or (predictions, lower_bound, upper_bound) if return_interval=True
        """
        X = np.array(X).reshape(-1, 1)
        X_poly = self.poly_features.transform(X)
        y_pred = self.model.predict(X_poly)
        
        if return_interval:
            # Prediction interval widens with distance from training data
            X_mean = np.mean(self.X_train)
            distance_factor = 1 + 0.1 * np.abs(X.flatten() - X_mean) / (np.std(self.X_train) + 1e-10)
            interval = z * self.residual_std * distance_factor
            return y_pred, y_pred - interval, y_pred + interval
        
        return y_pred
    
    def get_equation(self):
        """
        Return the fitted equation as a string.
        
        Returns
        -------
        str
            Formatted equation string
        """
        coefs = self.model.coef_
        if self.degree == 2:
            return f"C(V) = {coefs[0]:.2f} + {coefs[1]:.3f}V + {coefs[2]:.4f}V²"
        else:
            terms = [f"{coefs[0]:.2f}"]
            for i in range(1, len(coefs)):
                terms.append(f"{coefs[i]:.4f}V^{i}")
            return "C(V) = " + " + ".join(terms)
    
    def get_coefficients(self):
        """
        Return the model coefficients.
        
        Returns
        -------
        ndarray
            Array of polynomial coefficients [intercept, linear, quadratic, ...]
        """
        return self.model.coef_
