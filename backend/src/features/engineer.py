import polars as pl
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_date_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts Month, Weekday, and Weekend/Special Event flags.
    """
    logger.info("Creating Date Features...")
    return df.with_columns(
        [
            pl.col("date").dt.month().alias("month").cast(pl.Int8),
            pl.col("date").dt.weekday().alias("wday").cast(pl.Int8),
            pl.col("date").dt.year().alias("year").cast(pl.Int16),
            # Create a weekend flag (Saturday=6, Sunday=7)
            (pl.col("date").dt.weekday() >= 6).alias("is_weekend").cast(pl.Int8),
        ]
    )


def create_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates Lag and Rolling features.
    CRITICAL: We must group by 'id' (item_store) before shifting!
    """
    logger.info("Creating Lag & Rolling Features (This takes RAM)...")

    # We rely on the physical order of data being sorted by date per group
    # Polars 'over' window functions are perfect for this.

    # 1. Simple Lags (7 days ago, 28 days ago)
    # Note: Lag 28 is the "safe" lag for a 28-day forecast horizon without recursion
    lags = [7, 14, 21, 28]
    lag_cols = [
        pl.col("sales").shift(lag).over("id").alias(f"lag_{lag}") for lag in lags
    ]

    # 2. Rolling Features (Moving Averages)
    # We take the Lag-28 column and apply rolling windows on it to prevent leakage
    rolling_cols = [
        pl.col("sales")
        .shift(28)
        .rolling_mean(window_size=7)
        .over("id")
        .alias("rolling_mean_7"),
        pl.col("sales")
        .shift(28)
        .rolling_mean(window_size=28)
        .over("id")
        .alias("rolling_mean_28"),
    ]

    return df.with_columns(lag_cols + rolling_cols)


def create_price_features(df: pl.DataFrame, prices_df: pl.DataFrame) -> pl.DataFrame:
    """
    Joins price data and calculates price momentum.
    """
    logger.info("Joining Prices and creating Price Momentum...")

    # Join prices on item_id, store_id, and wm_yr_wk (which comes from calendar)
    # First we need to ensure df has wm_yr_wk.
    # Ideally, this merge happened in the melt step or we do it here via calendar.
    # For simplicity, we assume the 'melted_df' already has 'item_id', 'store_id', 'date'.
    # We need to join 'calendar' first to get 'wm_yr_wk' if it's not there.

    # NOTE: In our preprocess.py, we only melted. We didn't join calendar yet.
    # So we will do a robust join in the main script.
    # Here we assume df ALREADY has 'sell_price' from that join.

    return df.with_columns(
        [
            # Price Momentum: Current Price / Average Price of that item globally
            (pl.col("sell_price") / pl.col("sell_price").mean().over("item_id"))
            .alias("price_momentum")
            .cast(pl.Float32)
        ]
    )
