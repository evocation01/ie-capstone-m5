import sys
from pathlib import Path
import pytest
import numpy as np

# Add backend root to sys.path to import src scripts
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.optimization.optimize import calculate_metrics, calculate_costs

def test_calculate_metrics_perfect_forecast():
    """Test RMSE calculation when forecast is exactly the same as actuals."""
    forecast = np.array([[10, 20, 30, 40, 50, 60, 70] * 4])  # 28 days
    actual = np.array([[10, 20, 30, 40, 50, 60, 70] * 4])
    
    rmse, weekly_rmse = calculate_metrics(forecast, actual)
    
    assert rmse == 0.0
    assert len(weekly_rmse) == 4
    for w_rmse in weekly_rmse:
        assert w_rmse == 0.0

def test_calculate_metrics_constant_error():
    """Test RMSE calculation with a constant error of 5."""
    forecast = np.array([[15, 25, 35, 45, 55, 65, 75] * 4])  # 28 days
    actual = np.array([[10, 20, 30, 40, 50, 60, 70] * 4])
    
    rmse, weekly_rmse = calculate_metrics(forecast, actual)
    
    assert rmse == 5.0
    assert len(weekly_rmse) == 4
    for w_rmse in weekly_rmse:
        assert w_rmse == 5.0

def test_calculate_costs_holding_only():
    """Test cost calculation when forecast > actual (over-forecasting = holding costs)."""
    # Over forecast by 10 units
    forecast = np.array([[20]])
    actual = np.array([[10]])
    holding_cost_rate = 1.0
    stockout_cost_rate = 10.0
    
    h_cost, s_cost, _, _ = calculate_costs(forecast, actual, holding_cost_rate, stockout_cost_rate)
    
    assert h_cost == 10.0 * 1.0  # 10 units excess * $1 holding
    assert s_cost == 0.0         # No stockouts

def test_calculate_costs_stockout_only():
    """Test cost calculation when forecast < actual (under-forecasting = stockout costs)."""
    # Under forecast by 5 units
    forecast = np.array([[15]])
    actual = np.array([[20]])
    holding_cost_rate = 1.0
    stockout_cost_rate = 10.0
    
    h_cost, s_cost, _, _ = calculate_costs(forecast, actual, holding_cost_rate, stockout_cost_rate)
    
    assert h_cost == 0.0         # No excess inventory
    assert s_cost == 5.0 * 10.0  # 5 units missing * $10 stockout

def test_calculate_costs_mixed():
    """Test cost calculation across multiple items and days."""
    forecast = np.array([
        [10, 5],  # Item 1: Over by 2, Under by 5
        [8, 12]   # Item 2: Perfect, Over by 2
    ])
    actual = np.array([
        [8, 10],
        [8, 10]
    ])
    holding_cost_rate = 2.0
    stockout_cost_rate = 15.0
    
    h_cost, s_cost, _, _ = calculate_costs(forecast, actual, holding_cost_rate, stockout_cost_rate)
    
    # Expected Holding: (2 units on Item1_Day1) + (2 units on Item2_Day2) = 4 units
    expected_h = 4 * 2.0
    
    # Expected Stockout: (5 units on Item1_Day2) = 5 units
    expected_s = 5 * 15.0
    
    assert h_cost == expected_h
    assert s_cost == expected_s
