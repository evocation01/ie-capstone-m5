import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.features.engineer import (
    create_date_features,
    create_lag_features,
    create_price_features,
)
from src.models.lstm import M5LSTM
from src.utils.logger import get_logger

logger = get_logger("predict")

# --- GLOBAL SCALER STATS ---
# We will compute these ONCE from the full history and reuse them
SCALER_STATS = {}


def prepare_features(df, cal, prices):
    """
    Re-runs feature engineering on a specific dataframe subset.
    """
    if "wm_yr_wk" not in df.columns:
        df = df.join(cal, left_on="d", right_on="d", how="left")

    if "sell_price" not in df.columns:
        df = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df = create_date_features(df)
    df = create_price_features(df, prices)
    df = df.sort(["id", "date"])
    df = create_lag_features(df)

    return df


def clean_and_encode(df, is_training_stats=False):
    """
    Applies cleaning/encoding.
    Uses global SCALER_STATS for normalization to ensure consistency.
    """
    num_cols = [
        c for c, t in zip(df.columns, df.dtypes) if t in [pl.Float32, pl.Float64]
    ]
    for col in num_cols:
        df = df.with_columns(pl.col(col).fill_nan(0).fill_null(0))

    # Normalize using SAVED global stats
    for col in num_cols:
        if col != "sales":
            if is_training_stats:
                # Compute and save stats (only done once on full history)
                mean = df.select(pl.col(col).mean()).item()
                std = df.select(pl.col(col).std()).item()
                SCALER_STATS[col] = {"mean": mean, "std": std}

            # Apply stats
            if col in SCALER_STATS:
                stats = SCALER_STATS[col]
                if stats["std"] > 1e-8:
                    df = df.with_columns(
                        ((pl.col(col) - stats["mean"]) / stats["std"]).alias(col)
                    )
                else:
                    df = df.with_columns((pl.col(col) * 0).alias(col))

    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    for col in cat_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical().fill_null(0)
            )

    bool_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Boolean]
    if bool_cols:
        df = df.with_columns([pl.col(c).cast(pl.Int8).fill_null(0) for c in bool_cols])

    return df


def main():
    logger.info("🚀 Starting Inference Pipeline...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = paths.MODELS_DIR / "lstm_best.pt"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    data_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"
    price_path = paths.PROCESSED_DATA_DIR / "sell_prices.parquet"

    logger.info("Loading raw data history...")

    # Load full data for CA_1
    full_df = (
        pl.read_parquet(data_path)
        .filter(pl.col("store_id") == "CA_1")
        .with_columns(pl.col("sales").cast(pl.Float32))
    )

    cal = pl.read_parquet(cal_path)
    prices = pl.read_parquet(price_path).with_columns(
        [
            pl.col("store_id").cast(pl.Categorical),
            pl.col("item_id").cast(pl.Categorical),
        ]
    )

    unique_ids = full_df["id"].unique().to_list()
    items_count = len(unique_ids)
    logger.info(f"Forecasting for {items_count} items...")

    # --- STEP 1: CALCULATE GLOBAL STATS ---
    # We run feature prep on a large chunk of history just to calculate Mean/Std
    # This ensures our normalization during inference matches "global" reality
    logger.info("Calculating global normalization stats...")
    stats_sample = full_df.sample(n=100000)  # Sample for speed
    stats_sample = prepare_features(stats_sample, cal, prices)
    clean_and_encode(
        stats_sample, is_training_stats=True
    )  # This populates SCALER_STATS
    logger.info(f"Stats computed for: {list(SCALER_STATS.keys())}")

    # --- STEP 2: PREPARE LOOP ---
    horizon = 28
    last_day = 1913
    history = full_df.filter(
        pl.col("d").str.extract(r"(\d+)").cast(pl.Int32) > (last_day - 100)
    )

    # Run initial prep to get features and shape
    # We clone to avoid modifying the loop variable 'current_df' prematurely
    temp_df = prepare_features(history.clone(), cal, prices)
    temp_df = clean_and_encode(temp_df)  # Uses SCALER_STATS

    features = [
        c
        for c in temp_df.columns
        if c not in ["sales", "date", "id", "d", "wm_yr_wk", "weekday"]
    ]
    num_features = len(features)
    logger.info(f"Model Input Features: {num_features}")

    model = M5LSTM(num_features=num_features).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    forecasts = []
    current_df = history

    logger.info("Starting recursive loop...")
    for day in range(1, horizon + 1):
        target_d = f"d_{last_day + day}"
        print(f"Predicting {target_d}...", end="\r")

        rich_df = prepare_features(current_df, cal, prices)
        rich_df = clean_and_encode(rich_df)

        seq_len = 28
        data_matrix = rich_df.select(features).to_numpy().astype(np.float32)

        # Safety check for reshaping
        if len(data_matrix) % items_count != 0:
            rows_per_item = len(data_matrix) // items_count
            data_matrix = data_matrix[: items_count * rows_per_item]
        else:
            rows_per_item = len(data_matrix) // items_count

        X = data_matrix.reshape(items_count, rows_per_item, num_features)
        X_in = X[:, -seq_len:, :]

        X_tensor = torch.tensor(X_in).to(device)
        with torch.no_grad():
            preds_log = model(X_tensor).cpu().numpy()

        # Clip negatives
        preds_log = np.maximum(preds_log, 0)
        preds = np.expm1(preds_log)

        if day == 1:
            print(
                f"\nDEBUG DAY 1: Min {preds.min()}, Max {preds.max()}, Mean {preds.mean()}"
            )

        forecasts.append(preds)

        # Prepare new rows
        meta_cols = ["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
        meta_map = history.select(meta_cols).unique(subset=["id"])

        new_rows = pl.DataFrame(
            {
                "id": sorted(unique_ids),
                "d": [target_d] * items_count,
                "sales": preds.flatten(),
            },
            schema={"id": pl.String, "d": pl.String, "sales": pl.Float32},
        )

        new_rows = new_rows.join(meta_map, on="id", how="left")
        new_rows = new_rows.select(current_df.columns)
        current_df = pl.concat([current_df, new_rows], how="vertical")

    logger.info("\nSaving Forecast...")

    # Fix shape mismatch error: ensure forecasts list is not empty
    if not forecasts:
        logger.error("No forecasts generated!")
        return

    # Convert list of arrays to 2D array
    # forecasts is list of (3049,) arrays. len=28
    forecast_array = np.array(forecasts).T  # Shape: (3049, 28)

    logger.info(f"Forecast Shape: {forecast_array.shape}")

    out_df = pd.DataFrame(forecast_array, columns=[f"F{i}" for i in range(1, 29)])
    out_df["id"] = sorted(unique_ids)

    # Save submission file
    logger.info(f"Saving submission file...")
    out_path = paths.FORECASTS_DIR / "forecast_lstm.csv"
    submission_df.to_csv(out_path, index=False)
    logger.info(f"Submission file saved to {out_path}")
    out_df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
