import argparse
import sys
from pathlib import Path

# Add 'src' to python path so imports work when running from scripts/
sys.path.append(str(Path(__file__).parents[2]))

from src.config import paths
from src.data.ingestion import load_and_melt_sales, load_calendar, load_prices
from src.utils.logger import get_logger

logger = get_logger("preprocess")


def main(args):
    logger.info("Starting data preprocessing pipeline...")

    # 1. Process Sales Data (The Big One)
    # UPDATED: Changed filename to match what you have (sales_train_validation.csv)
    sales_raw = paths.RAW_DATA_DIR / "sales_train_validation.csv"
    sales_out = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"

    if not sales_raw.exists():
        logger.error(f"File not found: {sales_raw}")
        logger.error("Did you run the download script in 'backend/data/raw'?")
        sys.exit(1)

    load_and_melt_sales(sales_raw, sales_out)

    # 2. Convert Calendar & Prices to Parquet (Simple conversion)
    # This makes loading them much faster in future steps
    cal_raw = paths.RAW_DATA_DIR / "calendar.csv"
    cal_out = paths.PROCESSED_DATA_DIR / "calendar.parquet"
    if cal_raw.exists():
        cal_df = load_calendar(cal_raw)
        cal_df.write_parquet(cal_out)
        logger.info(f"Saved {cal_out}")

    price_raw = paths.RAW_DATA_DIR / "sell_prices.csv"
    price_out = paths.PROCESSED_DATA_DIR / "sell_prices.parquet"
    if price_raw.exists():
        price_df = load_prices(price_raw)
        price_df.write_parquet(price_out)
        logger.info(f"Saved {price_out}")

    logger.info("✅ Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 Data Preprocessing")
    args = parser.parse_args()
    main(args)
