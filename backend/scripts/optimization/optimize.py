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
    logger.info("🚀 Starting Full Optimization & Analysis (3 Models)...")

    # 1. Load Data
    lstm_path = paths.EXPERIMENTS_DIR / "forecast_lstm.csv"
    lgbm_path = paths.EXPERIMENTS_DIR / "forecast_lgbm.csv"
    raw_path = paths.RAW_DATA_DIR / "sales_train_validation.csv"

    if not lstm_path.exists() or not raw_path.exists():
        logger.error("Missing forecast or raw data files.")
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
    logger.info("Loading Ground Truth...")
    df_actual = pd.read_csv(raw_path)

    # The forecast corresponds to the last 28 columns (d_1886 to d_1913)
    actual_cols = [c for c in df_actual.columns if c.startswith("d_")][-28:]
    ground_truth = df_actual.set_index("id").loc[unique_ids, actual_cols].values

    # Load Naive Baseline (Last 28 Days History)
    # 28 days BEFORE the validation period
    naive_cols = [c for c in df_actual.columns if c.startswith("d_")][-56:-28]
    naive_forecast = df_actual.set_index("id").loc[unique_ids, naive_cols].values

    # Cost Parameters
    HOLDING_COST = 1.0
    STOCKOUT_COST = 10.0

    results = []

    # --- ANALYSIS 1: NAIVE BASELINE ---
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
            "Total_Cost": h_cost + s_cost,
        }
    )

    # --- ANALYSIS 2: LIGHTGBM ---
    if has_lgbm:
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
                "Total_Cost": h_cost + s_cost,
            }
        )

    # --- ANALYSIS 3: LSTM (DEEP LEARNING) ---
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
            "Total_Cost": h_cost + s_cost,
        }
    )

    # --- ANALYSIS 4: SENSITIVITY (Robustness Check) ---
    # Scenario: What if LSTM is 10% worse? (Add 10% noise)
    lstm_matrix = lstm_matrix.astype(float)
    np.random.seed(42)

    noise = np.random.normal(0, 0.1 * (np.mean(lstm_matrix) + 1e-6), lstm_matrix.shape)
    lstm_noisy = np.maximum(lstm_matrix + noise, 0)

    h_cost_bad, s_cost_bad = calculate_costs(
        lstm_noisy, ground_truth, HOLDING_COST, STOCKOUT_COST
    )
    total_bad = h_cost_bad + s_cost_bad

    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL COMPARISON REPORT")
    logger.info("=" * 80)
    logger.info(
        f"{'Model':<15} | {'RMSE':<8} | {'W1 RMSE':<8} | {'W4 RMSE':<8} | {'TOTAL COST ($)':<15}"
    )
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['Model']:<15} | {r['RMSE']:.4f}   | {r['W1_RMSE']:.4f}   | {r['W4_RMSE']:.4f}   | ${r['Total_Cost']:,.0f}"
        )

    logger.info("-" * 80)
    logger.info(
        f"LSTM (Noisy -10%)| ------     | ------     | ------     | ${total_bad:,.0f}"
    )
    logger.info("=" * 80)

    # Calculate Savings vs Naive
    naive_cost = results[0]["Total_Cost"]
    lstm_cost = results[-1]["Total_Cost"]
    savings = naive_cost - lstm_cost

    if savings > 0:
        logger.info(f"🏆 LSTM saves ${savings:,.0f} vs Naive Baseline.")
    else:
        logger.info(f"⚠️ LSTM is ${abs(savings):,.0f} more expensive than Naive.")

    # Save Results
    res_df = pd.DataFrame(results)
    out_path = paths.EXPERIMENTS_DIR / "final_results.csv"
    res_df.to_csv(out_path, index=False)
    logger.info(f"Saved final results to {out_path}")


if __name__ == "__main__":
    main()
