import sys
from pathlib import Path
from datetime import timedelta

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import polars as pl
import pandas as pd
from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("extract_sample_raw")

def main():
    logger.info("🚀 Extracting Raw Sales for Sample 10...")

    # 1. Get the list of 10 IDs from the existing forecast sample
    sample_forecast_path = paths.RESULTS_DIR / "forecasts" / "forecast_lgbm_sample10.csv"
    if not sample_forecast_path.exists():
        logger.error(f"Sample forecast file not found at {sample_forecast_path}")
        return

    # We use pandas here just to quickly grab the IDs, as the file is small CSV
    sample_ids_df = pd.read_csv(sample_forecast_path)
    target_ids = sample_ids_df["id"].tolist()
    logger.info(f"Found {len(target_ids)} IDs in {sample_forecast_path.name}")

    # 2. Load Historical Sales Data (Parquet)
    sales_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    if not sales_path.exists():
        logger.error(f"Processed sales data not found at {sales_path}")
        return
    
    logger.info("Loading melted sales data (this may take a moment)...")
    # Lazy load
    q_sales = pl.scan_parquet(sales_path)

    # 3. Load Calendar to get actual dates (for filtering last 3 months)
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"
    if not cal_path.exists():
        logger.error(f"Calendar data not found at {cal_path}")
        return
    
    q_cal = pl.scan_parquet(cal_path)

    # 4. Join and Filter
    # Filter sales for our specific IDs first to reduce data size
    q_filtered = q_sales.filter(pl.col("id").is_in(target_ids))

    # Join with calendar to get 'date'
    # melted_sales has 'd' (e.g., 'd_1'), calendar has 'd' and 'date'
    q_joined = q_filtered.join(q_cal, on="d", how="left")

    # Collect to find the max date
    df = q_joined.collect()

    if df.is_empty():
        logger.error("No data found for the specified IDs.")
        return

    max_date = df["date"].max()
    min_date = max_date - timedelta(days=90) # Approx 3 months

    logger.info(f"Filtering data from {min_date} to {max_date} (Last ~90 days)")

    # Filter for the time range
    df_recent = df.filter(pl.col("date") >= min_date)

    # 5. Pivot to Wide Format (Excel friendly)
    # Rows: id
    # Columns: date (or d)
    # Values: sales
    
    # We prefer 'date' as column headers for clarity, or 'd' if preferred. 
    # Let's use 'date' formatted as string YYYY-MM-DD
    df_recent = df_recent.with_columns(pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str"))
    
    # Pivot: values='sales', index='id', columns='date_str'
    df_wide = df_recent.pivot(
        values="sales",
        index="id",
        columns="date_str",
        aggregate_function="sum" # Should be unique per id/date, but sum is safe
    )

    # Sort rows to match the order in the input file if possible, or just alphabetically
    df_wide = df_wide.sort("id")

    # 6. Save
    out_path = paths.RESULTS_DIR / "forecasts" / "raw_sales_sample10.csv"
    logger.info(f"Saving raw sales data to {out_path}...")
    df_wide.write_csv(out_path)

    logger.info("✅ Done! You can now open the file in Excel to calculate RMSE.")

if __name__ == "__main__":
    main()
