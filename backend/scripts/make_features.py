import sys
from pathlib import Path

# Add the backend root to sys.path so imports work
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import polars as pl
from src.config import paths
from src.features.engineer import (
    create_date_features,
    create_lag_features,
    create_price_features,
)
from src.utils.logger import get_logger

logger = get_logger("make_features")


def main():
    logger.info("🚀 Starting Feature Engineering Pipeline...")

    # 1. Load Data
    sales_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"
    price_path = paths.PROCESSED_DATA_DIR / "sell_prices.parquet"

    # Load Sales (already has Categorical types from preprocess.py)
    df = pl.read_parquet(sales_path)

    # Load Calendar
    cal = pl.read_parquet(cal_path)

    # Load Prices AND Cast to match Sales types
    # FIX: Cast store_id and item_id to Categorical so they match 'df'
    prices = pl.read_parquet(price_path).with_columns(
        [
            pl.col("store_id").cast(pl.Categorical),
            pl.col("item_id").cast(pl.Categorical),
        ]
    )

    # 2. Merge Calendar (to get 'wm_yr_wk' for price join and date features)
    logger.info("Merging Calendar...")
    # We left join on 'd' (e.g., "d_1"). Calendar likely has 'd' as string.
    # Ensure 'd' types match if this fails too, but usually they are both strings.
    df = df.join(cal, left_on="d", right_on="d", how="left")

    # 3. Merge Prices
    logger.info("Merging Prices...")
    # Now both sides have Categorical store_id and item_id
    df = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # 4. Apply Feature Engineering
    df = create_date_features(df)
    # Pass the 'prices' dataframe if needed by logic, though we just merged it.
    # If create_price_features uses the merged columns, we might not need to pass 'prices'
    # unless it does independent lookups.
    # Let's update the function call or ensure logic uses the columns now present in 'df'.
    df = create_price_features(df, prices)

    # Lags require sorting!
    logger.info("Sorting by date for Lag calculations...")
    df = df.sort(["id", "date"])
    df = create_lag_features(df)

    # 5. Cleanup & Optimization
    # Drop rows where lags are null (the first 28 days of data)
    logger.info("Dropping NaNs from Lag creation...")
    df = df.drop_nulls(subset=["lag_28"])

    # Drop unnecessary string columns to save memory
    # We drop 'd' and 'wm_yr_wk' as they are no longer needed for joining
    df = df.drop(["d", "wm_yr_wk", "weekday"])

    # 6. Save
    out_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    logger.info(f"Saving Feature-Rich Dataset to {out_path}...")
    df.write_parquet(out_path)

    logger.info("✅ Feature Engineering Complete. Ready for Training.")


if __name__ == "__main__":
    main()
