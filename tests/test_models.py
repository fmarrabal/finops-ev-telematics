#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for FinOps Cloud Cost Forecasting.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from finops_forecasting import (
    load_finops_data,
    compute_metrics,
    PolynomialForecast,
)


class TestDataLoading(unittest.TestCase):
    """Test data loading functions."""
    
    def test_load_finops_data_returns_dataframes(self):
        """Test that load_finops_data returns two DataFrames."""
        pre_finops, post_finops = load_finops_data()
        self.assertIsInstance(pre_finops, pd.DataFrame)
        self.assertIsInstance(post_finops, pd.DataFrame)
    
    def test_pre_finops_has_correct_length(self):
        """Test that pre-FinOps data has 12 months."""
        pre_finops, _ = load_finops_data()
        self.assertEqual(len(pre_finops), 12)
    
    def test_post_finops_has_correct_length(self):
        """Test that post-FinOps data has 40 months."""
        _, post_finops = load_finops_data()
        self.assertEqual(len(post_finops), 40)
    
    def test_data_columns_exist(self):
        """Test that required columns exist."""
        pre_finops, post_finops = load_finops_data()
        
        pre_cols = ['month_index', 'date', 'cloud_cost_index', 'vehicles_index']
        for col in pre_cols:
            self.assertIn(col, pre_finops.columns)
        
        post_cols = ['month_index', 'date', 'cloud_cost_index', 'vehicles_index', 
                     'total_messages_index', 'cost_per_billion_msg']
        for col in post_cols:
            self.assertIn(col, post_finops.columns)


class TestMetrics(unittest.TestCase):
    """Test metrics computation."""
    
    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect prediction."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = compute_metrics(y_true, y_pred, "Test")
        
        self.assertEqual(metrics['R²'], 1.0)
        self.assertEqual(metrics['RMSE'], 0.0)
        self.assertEqual(metrics['MAE'], 0.0)
        self.assertEqual(metrics['MAPE (%)'], 0.0)
    
    def test_compute_metrics_returns_dict(self):
        """Test that compute_metrics returns a dictionary."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])
        
        metrics = compute_metrics(y_true, y_pred, "Test")
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('Model', metrics)
        self.assertIn('R²', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE (%)', metrics)
        self.assertIn('sMAPE (%)', metrics)


class TestPolynomialForecast(unittest.TestCase):
    """Test Polynomial Forecast model."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.array([1, 2, 3, 4, 5])
        self.y = np.array([1, 4, 9, 16, 25])  # y = x^2
    
    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        model = PolynomialForecast(degree=2)
        result = model.fit(self.X, self.y)
        self.assertIs(result, model)
    
    def test_predict_returns_array(self):
        """Test that predict returns numpy array."""
        model = PolynomialForecast(degree=2)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_predict_with_interval(self):
        """Test prediction with confidence interval."""
        model = PolynomialForecast(degree=2)
        model.fit(self.X, self.y)
        pred, lower, upper = model.predict(self.X, return_interval=True)
        
        self.assertEqual(len(pred), len(self.X))
        self.assertEqual(len(lower), len(self.X))
        self.assertEqual(len(upper), len(self.X))
        
        # Lower should be less than or equal to prediction
        self.assertTrue(np.all(lower <= pred))
        # Upper should be greater than or equal to prediction
        self.assertTrue(np.all(upper >= pred))


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline_runs(self):
        """Test that the full analysis pipeline runs without error."""
        pre_finops, post_finops = load_finops_data()
        
        # Fit polynomial model
        model = PolynomialForecast(degree=2)
        model.fit(
            pre_finops['vehicles_index'].values,
            pre_finops['cloud_cost_index'].values
        )
        
        # Make predictions
        predictions = model.predict(pre_finops['vehicles_index'].values)
        
        # Compute metrics
        metrics = compute_metrics(
            pre_finops['cloud_cost_index'].values,
            predictions,
            "Polynomial"
        )
        
        # Check R² is reasonable
        self.assertGreater(metrics['R²'], 0.9)


if __name__ == '__main__':
    unittest.main()
