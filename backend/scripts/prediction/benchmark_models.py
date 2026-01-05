import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.models.classical import ClassicalForecaster
from src.models.ml_baselines import ProphetForecaster, SimpleMLForecaster
from src.utils.logger import get_logger

logger = get_logger("benchmark_models")


def main():
    logger.info("🏎️  Starting All-Tier Algorithm Drag Race (Full History, 3049 items)!")

    # 1. Load the 3049 IDs from LSTM forecast (The "Evaluation Set")
    lstm_path = paths.FORECASTS_DIR / "forecast_lstm.csv"
    lgbm_path = paths.FORECASTS_DIR / "forecast_lgbm.csv"

    if not lstm_path.exists():
        logger.error(f"LSTM forecast not found at {lstm_path}")
        return

    target_ids = pd.read_csv(lstm_path)["id"].tolist()
    logger.info(f"Loaded {len(target_ids)} Target IDs for the race.")

    # 2. Load Tier 3 Forecasts (Pre-calculated)
    tier3_data = {}
    forecast_cols = [f"F{i}" for i in range(1, 29)]

    if lgbm_path.exists():
        logger.info("Loading LightGBM Forecasts...")
        lgbm_df = pd.read_csv(lgbm_path).set_index("id")
        tier3_data["LightGBM (Tier 3)"] = lgbm_df

    if lstm_path.exists():
        logger.info("Loading LSTM Forecasts...")
        lstm_df = pd.read_csv(lstm_path).set_index("id")
        tier3_data["LSTM (Tier 3)"] = lstm_df

    # 3. Load Full Historical Data (Parquet)
    sales_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"

    logger.info("Loading melted sales data (Lazy)...")
    q_sales = pl.scan_parquet(sales_path)
    q_cal = pl.scan_parquet(cal_path)

    # 4. Filter & Join
    q_filtered = q_sales.filter(pl.col("id").is_in(target_ids))
    q_joined = q_filtered.join(q_cal, on="d", how="left")
    df = q_joined.collect()
    df = df.sort(["id", "date"])

    # 5. Pivot for easy iteration (Wide Format)
    logger.info("Pivoting data for iteration...")
    df = df.with_columns(pl.col("date").dt.strftime("%Y-%m-%d"))
    df_wide = df.pivot(
        values="sales", index="id", columns="date", aggregate_function="sum"
    )

    pdf = df_wide.to_pandas().set_index("id")

    # 6. Train/Test Split
    HORIZON = 28
    date_cols = sorted(pdf.columns)
    train_cols = date_cols[:-HORIZON]
    test_cols = date_cols[-HORIZON:]

    logger.info(
        f"Training Range: {train_cols[0]} to {train_cols[-1]} ({len(train_cols)} days)"
    )
    logger.info(
        f"Testing Range:  {test_cols[0]} to {test_cols[-1]} ({len(test_cols)} days)"
    )

    results = []

    # Progress Bar
    try:
        from tqdm import tqdm

        iterator = tqdm(pdf.iterrows(), total=len(pdf), desc="Benchmarking")
    except ImportError:
        iterator = pdf.iterrows()
        logger.info("tqdm not found, using simple loop")

    # 7. Race Loop
    for item_id, row in iterator:
        y_train = row[train_cols].astype(float)
        y_test = row[test_cols].astype(float)
        y_train = y_train.fillna(0)

        # Optimization: Start training from first non-zero
        if (y_train != 0).any():
            y_vals = y_train.values
            # Fast numpy search for first non-zero
            start_loc = np.argmax(y_vals != 0)
            y_train_clean = y_train.iloc[start_loc:]
        else:
            # Skip items with no history
            continue

        # Initialize Pit Crew
        classical = ClassicalForecaster(y_train_clean, horizon=HORIZON)
        ml = SimpleMLForecaster(y_train_clean, horizon=HORIZON)
        # prophet_m = ProphetForecaster(y_train_clean, horizon=HORIZON)

        models = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Classical Models (Fast)
            models = {
                "Naive": classical.naive(),
                "SMA_28": classical.moving_average(window=28),
                "WMA_3": classical.weighted_moving_average(weights=[0.5, 0.3, 0.2]),
                "SES (alpha=0.2)": classical.exponential_smoothing(alpha=0.2),
                "Holt-Winters": classical.holt_winters(),
                "ETS (State Space)": classical.ets(),
                "ARIMA (1,1,1)": classical.arima(order=(1, 1, 1)),
                "XGBoost (Tier 2)": ml.xgboost(),
                # "Auto ARIMA (Tier 2)": classical.auto_arima(), # TOO SLOW for 3000 items
                # "Random Forest (Tier 2)": ml.random_forest(),  # Slower
                # "Prophet (Tier 2)": prophet_m.forecast()       # TOO SLOW
            }

        # Add Pre-Calculated Tier 3 Models
        for name, data_df in tier3_data.items():
            if item_id in data_df.index:
                try:
                    vals = data_df.loc[item_id, forecast_cols].values.astype(float)
                    models[name] = pd.Series(vals)
                except:
                    pass

        # Evaluation
        for model_name, y_pred in models.items():
            y_pred = y_pred.fillna(0)

            y_true_arr = y_test.values
            y_pred_arr = y_pred.values

            if len(y_pred_arr) != len(y_true_arr):
                y_pred_arr = y_pred_arr[: len(y_true_arr)]

            rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))

            results.append({"id": item_id, "model": model_name, "rmse": round(rmse, 4)})

    # 8. Finish Line
    results_df = pd.DataFrame(results)

    out_dir = paths.RESULTS_DIR / "benchmark"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "benchmark_results_full.csv"
    results_df.to_csv(out_path, index=False)

    # 9. Leaderboard
    leaderboard = results_df.groupby("model")["rmse"].mean().sort_values()

    print("\n" + "=" * 50)
    print("🏆  FINAL BENCHMARK LEADERBOARD (Avg RMSE)  🏆")
    print("=" * 50)
    print(leaderboard)
    print("=" * 50)

    logger.info(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
