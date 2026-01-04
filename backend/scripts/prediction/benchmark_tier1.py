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
from src.utils.logger import get_logger

logger = get_logger("benchmark_tier1")


def main():
    logger.info("🏎️  Starting Tier 1 & 2 Algorithm Drag Race (Full History)!")

    # 1. Load the 10 Sample IDs
    # We still want to race on the same 10 items for comparison,
    # but using their FULL history.
    sample_path = paths.FORECASTS_DIR / "forecast_lgbm_sample10.csv"
    if not sample_path.exists():
        logger.error(f"Sample IDs not found at {sample_path}")
        return

    target_ids = pd.read_csv(sample_path)["id"].tolist()
    logger.info(f"Loaded {len(target_ids)} Target IDs for the race.")

    # 2. Load Full Historical Data (Parquet)
    sales_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"

    logger.info("Loading melted sales data (Lazy)...")
    q_sales = pl.scan_parquet(sales_path)
    q_cal = pl.scan_parquet(cal_path)

    # 3. Filter & Join
    # Filter for our 10 racers
    q_filtered = q_sales.filter(pl.col("id").is_in(target_ids))

    # Join with Calendar to get real dates
    q_joined = q_filtered.join(q_cal, on="d", how="left")

    # Collect
    df = q_joined.collect()

    # Sort by date
    df = df.sort(["id", "date"])

    # 4. Pivot for easy iteration (Wide Format)
    # We want a DataFrame where index=id, columns=date, values=sales
    # This might be memory intensive for 30k items, but for 10 items it's trivial.
    # Note: Polars pivot is powerful.

    logger.info("Pivoting data for iteration...")
    # Ensure date is string for column names
    df = df.with_columns(pl.col("date").dt.strftime("%Y-%m-%d"))

    df_wide = df.pivot(
        values="sales", index="id", columns="date", aggregate_function="sum"
    )

    # Convert to Pandas for the "Pit Crew" (ClassicalForecaster)
    pdf = df_wide.to_pandas().set_index("id")

    # 5. Train/Test Split
    # Last 28 days = Test
    # Rest = Train
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

    # 6. Race Loop
    for item_id, row in pdf.iterrows():
        # row is the full history for one item
        y_train = row[train_cols].astype(float)
        y_test = row[test_cols].astype(float)

        # Fill NaNs in history (e.g. before item was sold) with 0
        y_train = y_train.fillna(0)

        # Optimization: Start training only after the first non-zero sale?
        # For simple models, 0s can skew things like Moving Average.
        # Let's strip leading zeros for cleaner training.
        if (y_train != 0).any():
            first_sale_idx = (y_train != 0).idxmax()
            # Slice from first sale to end of training
            # Pandas indexing with labels is inclusive
            start_loc = y_train.index.get_loc(first_sale_idx)
            y_train_clean = y_train.iloc[start_loc:]
        else:
            logger.warning(f"Item {item_id} has no sales in training period.")
            continue

        # Initialize Pit Crew
        forecaster = ClassicalForecaster(y_train_clean, horizon=HORIZON)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            models = {
                "Naive": forecaster.naive(),
                "SMA_28": forecaster.moving_average(window=28),
                "WMA_3": forecaster.weighted_moving_average(weights=[0.5, 0.3, 0.2]),
                "SES (alpha=0.2)": forecaster.exponential_smoothing(alpha=0.2),
                "Holt Linear": forecaster.holt_linear(),
                "Holt-Winters": forecaster.holt_winters(),
                "ARIMA (1,1,1)": forecaster.arima(order=(1,1,1)),
                "Auto ARIMA (Tier 2)": forecaster.auto_arima()
            }

        for model_name, y_pred in models.items():
            y_pred = y_pred.fillna(0)
            if len(y_pred) != len(y_test):
                y_pred = y_pred[: len(y_test)]

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append({"id": item_id, "model": model_name, "rmse": round(rmse, 4)})

    # 7. Finish Line
    results_df = pd.DataFrame(results)

    # Save detailed results
    out_dir = paths.RESULTS_DIR / "benchmark"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "benchmark_tier1_results.csv"
    results_df.to_csv(out_path, index=False)

    # 8. Leaderboard
    leaderboard = results_df.groupby("model")["rmse"].mean().sort_values()

    print("\n" + "=" * 40)
    print("🏆  BENCHMARK LEADERBOARD (Avg RMSE)  🏆")
    print("=" * 40)
    print(leaderboard)
    print("=" * 40)

    logger.info(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
