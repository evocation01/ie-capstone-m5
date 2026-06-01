import pytest
import numpy as np
import pandas as pd
from src.models.classical import ClassicalForecaster
from src.models.baseline import BaselineModel

def test_classical_forecaster_holt_winters():
    """Test Holt-Winters model on dummy data."""
    # Create some dummy data with a seasonal pattern
    np.random.seed(42)
    time = np.arange(100)
    data = 10 + 0.1 * time + 5 * np.sin(2 * np.pi * time / 7) + np.random.normal(0, 0.5, 100)
    history = pd.Series(data)
    
    forecaster = ClassicalForecaster(history, horizon=7)
    forecast = forecaster.holt_winters()
    
    # Assert forecast is generated for the right horizon
    assert len(forecast) == 7
    # Assert there are no NaNs in forecast
    assert not np.isnan(forecast).any()

def test_baseline_lightgbm():
    """Test baseline LightGBM model initialization."""
    model = BaselineModel()
    assert model.params["objective"] == "regression"
    assert model.model is None

def test_classical_moving_average():
    """Test moving average model on dummy data."""
    data = [10, 20, 30, 40, 50, 60, 70]
    history = pd.Series(data)
    
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.moving_average(window=3)
    
    assert len(forecast) == 2
    assert not np.isnan(forecast).any()

def test_classical_weighted_moving_average():
    """Test weighted moving average model on dummy data."""
    data = [10, 20, 30, 40, 50, 60, 70]
    history = pd.Series(data)
    
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.weighted_moving_average(weights=[0.5, 0.3, 0.2])
    
    assert len(forecast) == 2
    assert not np.isnan(forecast).any()
