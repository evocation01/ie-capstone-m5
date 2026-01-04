import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class SimpleMLForecaster:
    """
    Implements Tier 2 Machine Learning models (XGBoost, Random Forest)
    using simple lag-based features.
    """

    def __init__(self, history: pd.Series, horizon: int = 28):
        self.history = history.copy()
        self.horizon = horizon

    def _prepare_features(self, series: pd.Series):
        """Creates simple lag and rolling features for a single series."""
        df = pd.DataFrame(series)
        df.columns = ["target"]

        # Lags
        for lag in [1, 7, 14, 28]:
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # Rolling means
        df["roll_mean_7"] = df["target"].shift(1).rolling(window=7).mean()
        df["roll_mean_28"] = df["target"].shift(1).rolling(window=28).mean()

        return df.dropna()

    def _recursive_forecast(self, model):
        """Forecasting step-by-step to handle lags of predicted values."""
        current_series = self.history.tolist()
        predictions = []

        for _ in range(self.horizon):
            # Use same feature logic as training
            # We must use the values at the tail of current_series
            row = {
                "lag_1": current_series[-1],
                "lag_7": current_series[-7] if len(current_series) >= 7 else 0,
                "lag_14": current_series[-14] if len(current_series) >= 14 else 0,
                "lag_28": current_series[-28] if len(current_series) >= 28 else 0,
                "roll_mean_7": (
                    np.mean(current_series[-7:]) if len(current_series) >= 7 else 0
                ),
                "roll_mean_28": (
                    np.mean(current_series[-28:]) if len(current_series) >= 28 else 0
                ),
            }

            X_next = pd.DataFrame([row])
            pred = model.predict(X_next)[0]
            pred = max(0, pred)  # No negative sales

            predictions.append(pred)
            current_series.append(pred)

        return pd.Series(predictions)

    def xgboost(self) -> pd.Series:
        """XGBoost Regressor (Tier 2)"""
        df = self._prepare_features(self.history)
        if df.empty:
            return pd.Series([0] * self.horizon)

        X = df.drop(columns=["target"])
        y = df["target"]

        model = XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, verbosity=0
        )
        model.fit(X, y)

        return self._recursive_forecast(model)

    def random_forest(self) -> pd.Series:
        """Random Forest Regressor (Tier 2)"""
        df = self._prepare_features(self.history)
        if df.empty:
            return pd.Series([0] * self.horizon)

        X = df.drop(columns=["target"])
        y = df["target"]

        model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        model.fit(X, y)

        return self._recursive_forecast(model)


class ProphetForecaster:
    """Facebook Prophet (Tier 2)"""

    def __init__(self, history: pd.Series, horizon: int = 28):
        # Prophet expects columns 'ds' (datestamp) and 'y' (value)
        self.df = history.reset_index()
        self.df.columns = ["ds", "y"]
        self.horizon = horizon

    def forecast(self) -> pd.Series:
        # Disable logging to keep benchmark clean
        logging.getLogger("prophet").setLevel(logging.ERROR)

        try:
            # We disable seasonality/daily for speed on the small benchmark
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            model.fit(self.df)

            future = model.make_future_dataframe(periods=self.horizon)
            forecast = model.predict(future)

            return forecast["yhat"].tail(self.horizon).reset_index(drop=True)
        except:
            # Fallback to naive
            return pd.Series([self.df["y"].iloc[-1]] * self.horizon)
