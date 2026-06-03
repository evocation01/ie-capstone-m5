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

def test_classical_naive():
    """Test naive model on dummy data."""
    data = [10, 20, 30]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.naive()
    assert len(forecast) == 2
    assert (forecast == 30).all()

def test_classical_exponential_smoothing():
    data = [10, 12, 14, 16, 18, 20]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.exponential_smoothing()
    assert len(forecast) == 2

def test_classical_holt_linear():
    data = [10, 12, 14, 16, 18, 20]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.holt_linear()
    assert len(forecast) == 2

def test_classical_arima():
    data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.arima(order=(1,0,0))
    assert len(forecast) == 2

def test_classical_ets():
    data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    forecast = forecaster.ets()
    assert len(forecast) == 2

def test_baseline_train_predict_save_load(tmp_path):
    """Test full pipeline of BaselineModel (LightGBM)."""
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 5)
    y_val = np.random.rand(20)

    # Train
    model = BaselineModel({"objective": "regression", "verbose": -1})
    model.train(X_train, y_train, X_val, y_val)
    assert model.model is not None

    # Predict
    preds = model.predict(X_val)
    assert len(preds) == 20

    # Save & Load
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    assert save_path.exists()

    new_model = BaselineModel()
    new_model.load(save_path)
    assert new_model.model is not None
    
    # Predict with loaded model
    # Predict with loaded model
    loaded_preds = new_model.predict(X_val)
    np.testing.assert_array_almost_equal(preds, loaded_preds)

def test_classical_auto_arima():
    data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    # This might be slow, so we provide dummy data that's perfectly linear
    forecast = forecaster.auto_arima()
    assert len(forecast) == 2

def test_classical_fallbacks():
    """Test the exception blocks and fallbacks with insufficient data."""
    # Data too short to run most complex models
    data = [10]
    history = pd.Series(data)
    forecaster = ClassicalForecaster(history, horizon=2)
    
    # WMA fallback
    forecast_wma = forecaster.weighted_moving_average(weights=[0.5, 0.3, 0.2])
    assert len(forecast_wma) == 2

    # Provide completely broken data (string) to trigger except blocks
    broken_history = pd.Series(["a", "b", "c"])
    broken_forecaster = ClassicalForecaster(broken_history, horizon=2)

    # Exponential smoothing fallback
    assert len(broken_forecaster.exponential_smoothing()) == 2
    
    # Holt Linear fallback
    assert len(broken_forecaster.holt_linear()) == 2
    
    # Holt Winters fallback
    assert len(broken_forecaster.holt_winters()) == 2
    
    # ARIMA fallback
    assert len(broken_forecaster.arima()) == 2
    
    # Auto ARIMA fallback
    assert len(broken_forecaster.auto_arima()) == 2
    
    # ETS fallback
    assert len(broken_forecaster.ets()) == 2
