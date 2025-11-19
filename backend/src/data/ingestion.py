from pathlib import Path

import polars as pl
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_melt_sales(
    input_path: Path, output_path: Path, save_parquet: bool = True
) -> pl.DataFrame:
    """
    Loads the M5 sales data, melts it from wide to long format,
    casts types for memory efficiency, and optionally saves to Parquet.
    """
    logger.info(f"Loading raw data from: {input_path}")

    # Lazy load for efficiency
    # We scan_csv to create a lazyframe, allowing optimization before execution
    q = pl.scan_csv(input_path)

    # 1. Identification columns (Keep these fixed)
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # 2. Melt operation
    # The columns d_1, d_2, ... d_1941 need to become rows
    # We select the id vars and all columns starting with 'd_'
    logger.info("Melting dataframe... this might take a moment.")

    q_melted = q.melt(
        id_vars=id_vars, value_name="sales", variable_name="d"
    ).with_columns(
        [
            # Cast categorical columns to save massive amounts of RAM
            pl.col("item_id").cast(pl.Categorical),
            pl.col("dept_id").cast(pl.Categorical),
            pl.col("cat_id").cast(pl.Categorical),
            pl.col("store_id").cast(pl.Categorical),
            pl.col("state_id").cast(pl.Categorical),
            # Sales can be int16 (max 32,767 is enough for daily item sales)
            pl.col("sales").cast(pl.Int16),
        ]
    )

    # Execute the plan
    df = q_melted.collect()
    logger.info(f"Data melted. Shape: {df.shape}")

    if save_parquet:
        logger.info(f"Saving to Parquet: {output_path}")
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        logger.info("Save complete.")

    return df


def load_calendar(path: Path) -> pl.DataFrame:
    """
    Loads calendar.csv and parses dates.
    """
    logger.info(f"Loading calendar from: {path}")
    df = pl.read_csv(path).with_columns(pl.col("date").str.to_date())
    return df


def load_prices(path: Path) -> pl.DataFrame:
    """
    Loads sell_prices.csv.
    """
    logger.info(f"Loading prices from: {path}")
    df = pl.read_csv(path)
    return df
