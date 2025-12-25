#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Neural Network model for time series forecasting.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class LSTMForecast:
    """
    LSTM (Long Short-Term Memory) neural network for time series forecasting.
    
    This implementation includes trend-guided extrapolation and soft bounds
    to prevent explosive predictions during recursive forecasting.
    
    Parameters
    ----------
    lookback : int, default=3
        Number of previous time steps to use as input
    lstm_units : list, default=[50, 30]
        Number of units in each LSTM layer
    dropout_rate : float, default=0.1
        Dropout rate for regularization
    learning_rate : float, default=0.005
        Learning rate for Adam optimizer
    epochs : int, default=300
        Maximum number of training epochs
    batch_size : int, default=2
        Training batch size
    patience : int, default=30
        Early stopping patience
        
    Attributes
    ----------
    model : Sequential
        Keras LSTM model
    scaler : MinMaxScaler
        Scaler for input normalization
    history : History
        Training history
        
    Example
    -------
    >>> model = LSTMForecast(lookback=3)
    >>> model.fit(cost_series)
    >>> forecast = model.forecast(steps=12)
    """
    
    def __init__(self, lookback=3, lstm_units=[50, 30], dropout_rate=0.1,
                 learning_rate=0.005, epochs=300, batch_size=2, patience=30):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        
        self.model = None
        self.scaler = None
        self.history = None
        self.y_train = None
        
    def _create_sequences(self, data):
        """Create input sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
    
    def _build_model(self):
        """Build the LSTM neural network architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.lstm_units[0], 
            activation='tanh',
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.lookback, 1)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(LSTM(units, activation='tanh', return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def fit(self, y, verbose=0):
        """
        Fit the LSTM model to time series data.
        
        Parameters
        ----------
        y : array-like
            Time series values
        verbose : int, default=0
            Verbosity level for training
            
        Returns
        -------
        self
            Fitted model instance
        """
        self.y_train = np.array(y).flatten()
        
        # Normalize data
        self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        y_scaled = self.scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y_seq = self._create_sequences(y_scaled)
        X = X.reshape(-1, self.lookback, 1)
        
        # Build and train model
        self.model = self._build_model()
        
        early_stop = EarlyStopping(
            monitor='loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self
    
    def get_fitted_values(self):
        """
        Return in-sample fitted values.
        
        Returns
        -------
        ndarray
            Fitted values (length = original length - lookback)
        """
        y_scaled = self.scaler.transform(self.y_train.reshape(-1, 1)).flatten()
        X, _ = self._create_sequences(y_scaled)
        X = X.reshape(-1, self.lookback, 1)
        
        y_pred_scaled = self.model.predict(X, verbose=0).flatten()
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def forecast(self, steps, trend_weight_start=0.3, trend_weight_max=0.6):
        """
        Generate forecasts with trend-guided extrapolation.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        trend_weight_start : float, default=0.3
            Initial weight for trend blending
        trend_weight_max : float, default=0.6
            Maximum weight for trend blending
            
        Returns
        -------
        dict
            Dictionary with 'pred', 'lower', and 'upper' keys
        """
        y_scaled = self.scaler.transform(self.y_train.reshape(-1, 1)).flatten()
        
        # Initialize sequence with last lookback values
        current_seq = y_scaled[-self.lookback:]
        
        # Compute trend for guidance
        trend_slope = np.polyfit(np.arange(len(self.y_train)), self.y_train, 1)[0]
        
        predictions = []
        for step in range(steps):
            # Predict next value
            X_pred = current_seq.reshape(1, self.lookback, 1)
            next_val_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            next_val = self.scaler.inverse_transform([[next_val_scaled]])[0, 0]
            
            # Blend with trend
            expected_val = self.y_train[-1] + trend_slope * (step + 1)
            trend_weight = min(trend_weight_start + step * 0.015, trend_weight_max)
            next_val = (1 - trend_weight) * next_val + trend_weight * expected_val
            
            # Apply soft bounds
            max_allowed = self.y_train.max() + (self.y_train.max() - self.y_train.min()) * (step + 1) * 0.12
            next_val = np.clip(next_val, self.y_train.min() * 0.5, max_allowed)
            
            predictions.append(next_val)
            
            # Update sequence
            next_norm = self.scaler.transform([[next_val]])[0, 0]
            next_norm = np.clip(next_norm, 0.1, 0.9)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = next_norm
        
        predictions = np.array(predictions)
        
        # Compute prediction intervals
        std_est = np.std(self.y_train) * 0.3
        horizon_factor = np.sqrt(np.arange(1, steps + 1)) / 3
        interval = 1.96 * std_est * horizon_factor
        
        return {
            'pred': predictions,
            'lower': predictions - interval,
            'upper': predictions + interval
        }
