import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("optimize")


def calculate_metrics(forecast, actual):
    """Calculates RMSE and Weekly RMSEs."""
    # Ensure inputs are numpy arrays of floats
    forecast = np.array(forecast, dtype=float)
    actual = np.array(actual, dtype=float)

    mse = np.mean((forecast - actual) ** 2)
    rmse = np.sqrt(mse)

    # Weekly RMSE
    weeks = 4
    weekly_rmse = []
    days_per_week = 7

    for w in range(weeks):
        start = w * days_per_week
        end = (w + 1) * days_per_week

        # Handle shapes (Items, 28) -> (Items, 7)
        if start >= forecast.shape[1]:
            break

        w_pred = forecast[:, start:end]
        w_act = actual[:, start:end]
        w_rmse = np.sqrt(np.mean((w_pred - w_act) ** 2))
        weekly_rmse.append(w_rmse)

    return rmse, weekly_rmse


def calculate_costs(forecast_matrix, actual_matrix, holding_cost, stockout_cost):
    """
    Calculates financial costs (Holding + Stockout).
    """
    stock_levels = np.array(forecast_matrix, dtype=float)
    actuals = np.array(actual_matrix, dtype=float)

    diff = stock_levels - actuals

    # Positive diff = Leftover Stock (Holding Cost)
    holding_matrix = np.maximum(diff, 0) * holding_cost

    # Negative diff = Missed Sales (Stockout Cost)
    stockout_matrix = np.maximum(-diff, 0) * stockout_cost

    return np.sum(holding_matrix), np.sum(stockout_matrix)


def main():
    logger.info("🚀 Starting Optimization Analysis...")

    # 1. Load Data
    lstm_path = paths.FORECASTS_DIR / "forecast_lstm.csv"
    lgbm_path = paths.FORECASTS_DIR / "forecast_lgbm.csv"
    raw_path = paths.RAW_DATA_DIR / "sales_train_validation.csv"

    if not lstm_path.exists() or not raw_path.exists():
        logger.error(
            "Missing LSTM forecast or raw data files. Run prediction scripts first."
        )
        return

    # Load LSTM Forecast (The Anchor)
    df_lstm = pd.read_csv(lstm_path)
    unique_ids = df_lstm["id"].tolist()

    # Extract matrix
    f_cols = [c for c in df_lstm.columns if c.startswith("F")]
    lstm_matrix = df_lstm[f_cols].values

    # Load LGBM (Optional check)
    has_lgbm = lgbm_path.exists()
    if has_lgbm:
        logger.info("Loading LightGBM Forecast...")
        df_lgbm = pd.read_csv(lgbm_path)
        # Align with LSTM IDs
        df_lgbm = df_lgbm.set_index("id").reindex(unique_ids).reset_index()
        lgbm_matrix = df_lgbm[f_cols].values
    else:
        logger.warning("LightGBM forecast not found. Skipping LGBM comparison.")

    # Load Ground Truth
    logger.info("Loading Ground Truth for validation period...")
    df_actual = pd.read_csv(raw_path)

    # The forecast corresponds to the last 28 columns (d_1886 to d_1913)
    actual_cols = [c for c in df_actual.columns if c.startswith("d_")][-28:]
    ground_truth = df_actual.set_index("id").loc[unique_ids, actual_cols].values

    # Load Naive Baseline (Last 28 Days History)
    # 28 days BEFORE the validation period
    naive_cols = [c for c in df_actual.columns if c.startswith("d_")][-56:-28]
    naive_forecast = df_actual.set_index("id").loc[unique_ids, naive_cols].values

    # Cost Parameters
    HOLDING_COST = 1.0  # Simplified cost per unit of excess inventory
    STOCKOUT_COST = 10.0  # Simplified cost per unit of missed sales

    results = []

    # --- ANALYSIS 1: NAIVE BASELINE ---
    logger.info("Analyzing Naive (Last-Period) Baseline...")
    rmse_naive, weekly_naive = calculate_metrics(naive_forecast, ground_truth)
    h_cost, s_cost = calculate_costs(
        naive_forecast, ground_truth, HOLDING_COST, STOCKOUT_COST
    )
    results.append(
        {
            "Model": "Naive",
            "RMSE": rmse_naive,
            "W1_RMSE": weekly_naive[0],
            "W4_RMSE": weekly_naive[3],
            "Holding_Cost": h_cost,
            "Stockout_Cost": s_cost,
            "Total_Cost": h_cost + s_cost,
        }
    )

    # --- ANALYSIS 2: LIGHTGBM ---
    if has_lgbm:
        logger.info("Analyzing LightGBM Model...")
        rmse_lgbm, weekly_lgbm = calculate_metrics(lgbm_matrix, ground_truth)
        h_cost, s_cost = calculate_costs(
            lgbm_matrix, ground_truth, HOLDING_COST, STOCKOUT_COST
        )
        results.append(
            {
                "Model": "LightGBM",
                "RMSE": rmse_lgbm,
                "W1_RMSE": weekly_lgbm[0],
                "W4_RMSE": weekly_lgbm[3],
                "Holding_Cost": h_cost,
                "Stockout_Cost": s_cost,
                "Total_Cost": h_cost + s_cost,
            }
        )

    # --- ANALYSIS 3: LSTM (DEEP LEARNING) ---
    logger.info("Analyzing LSTM Model...")
    rmse_lstm, weekly_lstm = calculate_metrics(lstm_matrix, ground_truth)
    h_cost, s_cost = calculate_costs(
        lstm_matrix, ground_truth, HOLDING_COST, STOCKOUT_COST
    )
    results.append(
        {
            "Model": "LSTM",
            "RMSE": rmse_lstm,
            "W1_RMSE": weekly_lstm[0],
            "W4_RMSE": weekly_lstm[3],
            "Holding_Cost": h_cost,
            "Stockout_Cost": s_cost,
            "Total_Cost": h_cost + s_cost,
        }
    )

    # --- FINAL REPORT ---
    logger.info("\n" + "=" * 80)
    logger.info("📊 MODEL COMPARISON REPORT (COST & ACCURACY)")
    logger.info("=" * 80)
    report_str = (
        f"{'Model':<15} | {'RMSE':<8} | {'Total Cost':<15} | "
        f"{'Holding Cost':<15} | {'Stockout Cost':<15}"
    )
    logger.info(report_str)
    logger.info("-" * 80)

    for r in results:
        report_line = (
            f"{r['Model']:<15} | {r['RMSE']:.4f}   | ${r['Total_Cost']:<14,.0f} | "
            f"${r['Holding_Cost']:<14,.0f} | ${r['Stockout_Cost']:<14,.0f}"
        )
        logger.info(report_line)

    logger.info("=" * 80)

    # Calculate Savings vs Naive
    try:
        naive_cost = next(r["Total_Cost"] for r in results if r["Model"] == "Naive")
        lstm_cost = next(r["Total_Cost"] for r in results if r["Model"] == "LSTM")
        savings = naive_cost - lstm_cost
        percentage_savings = (savings / naive_cost) * 100 if naive_cost > 0 else 0

        logger.info(
            f"🏆 LSTM financial impact vs. Naive Baseline: ${savings:,.0f} ({percentage_savings:.2f}%)"
        )
    except StopIteration:
        logger.warning("Could not calculate financial impact due to missing models.")

    # Save Results
    res_df = pd.DataFrame(results)
    out_path = paths.OPTIMIZATION_DIR / "optimization_summary.csv"
    res_df.to_csv(out_path, index=False)
    logger.info(f"Saved optimization summary to {out_path}")


if __name__ == "__main__":
    main()
