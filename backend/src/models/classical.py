import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

class ClassicalForecaster:
    """
    Implements Tier 1 (Excel-like) and Tier 2 (ARIMA) forecasting methods.
    """
    
    def __init__(self, history: pd.Series, horizon: int = 28):
        self.history = history
        self.horizon = horizon

    def naive(self) -> pd.Series:
        """Forecast = Last observed value"""
        last_val = self.history.iloc[-1]
        return pd.Series([last_val] * self.horizon)

    def moving_average(self, window: int = 28) -> pd.Series:
        """Simple Moving Average of the last 'window' days"""
        ma = self.history.iloc[-window:].mean()
        return pd.Series([ma] * self.horizon)

    def weighted_moving_average(self, weights: list = [0.5, 0.3, 0.2]) -> pd.Series:
        """
        Weighted Moving Average. 
        Weights should sum to 1 and be ordered from most recent to oldest.
        Example: [0.5 (t-1), 0.3 (t-2), 0.2 (t-3)]
        """
        window = len(weights)
        # Grab last 'window' values. 
        # History is [... t-3, t-2, t-1]. 
        recent = self.history.iloc[-window:]
        
        # If weights are [0.5, 0.3, 0.2] (Most recent to oldest)
        # We need to reverse 'recent' to align: [t-1, t-2, t-3]
        recent_reversed = recent.iloc[::-1]
        
        if len(recent) < window:
             # Fallback to simple mean if not enough data
            return self.moving_average(window)
        
        wma = np.dot(recent_reversed.values, weights)
        return pd.Series([wma] * self.horizon)

    def exponential_smoothing(self, alpha: float = 0.2) -> pd.Series:
        """Single Exponential Smoothing (SES)"""
        try:
            model = SimpleExpSmoothing(self.history, initialization_method="estimated").fit(smoothing_level=alpha, optimized=False)
            return model.forecast(self.horizon)
        except:
            return self.naive()

    def holt_linear(self) -> pd.Series:
        """Holt's Linear Trend method"""
        try:
            model = ExponentialSmoothing(self.history, trend="add", seasonal=None).fit()
            return model.forecast(self.horizon)
        except:
            return self.naive()

    def holt_winters(self, seasonal_periods: int = 7) -> pd.Series:
        """Holt-Winters (Triple Exponential Smoothing) with Additive Seasonality"""
        try:
            # We need enough data for seasonality
            if len(self.history) < 2 * seasonal_periods:
                return self.holt_linear()
            
            model = ExponentialSmoothing(
                self.history, 
                trend="add", 
                seasonal="add", 
                seasonal_periods=seasonal_periods
            ).fit()
            return model.forecast(self.horizon)
        except:
            return self.holt_linear()

    def arima(self, order=(1, 1, 1)) -> pd.Series:
        """
        ARIMA Model (Tier 2).
        Default order (1,1,1) is a basic starting point. 
        In a real scenario, you'd use auto_arima to find the best order.
        """
        try:
            model = ARIMA(self.history, order=order).fit()
            return model.forecast(self.horizon)
        except:
            return self.naive()

    def auto_arima(self) -> pd.Series:
        """
        Auto ARIMA using pmdarima (Tier 2).
        Automatically selects the best (p,d,q) parameters based on AIC.
        """
        try:
            # We use a seasonal=True with m=7 for weekly seasonality
            # stepwise=True makes it faster
            model = pm.auto_arima(
                self.history, 
                seasonal=True, 
                m=7, 
                stepwise=True, 
                suppress_warnings=True,
                error_action="ignore",
                max_p=3, max_q=3 # Limiting for speed in benchmark
            )
            forecast = model.predict(n_periods=self.horizon)
            return pd.Series(forecast)
        except:
            # Fallback to simple ARIMA if auto fails
            return self.arima()
