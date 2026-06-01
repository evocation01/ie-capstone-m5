import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.features.engineer import (
    create_date_features,
    create_lag_features,
    create_price_features,
)
from src.utils.logger import get_logger

logger = get_logger("predict_lgbm")

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

def main():
    logger.info("🚀 Starting LightGBM Recursive Inference Pipeline...")

    # 1. Load Model
    model_path = paths.MODELS_DIR / "baseline_lgbm.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    model_features = model.feature_name()

    # 2. Load Data
    data_path = paths.PROCESSED_DATA_DIR / "melted_sales.parquet"
    cal_path = paths.PROCESSED_DATA_DIR / "calendar.parquet"
    price_path = paths.PROCESSED_DATA_DIR / "sell_prices.parquet"

    logger.info("Loading raw data history for all 30,490 SKUs...")
    
    full_df = pl.read_parquet(data_path)
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

    horizon = 28
    last_day = 1913
    
    # Keep last 100 days of history for lag calculations
    history = full_df.filter(
        pl.col("d").str.extract(r"(\d+)").cast(pl.Int32) > (last_day - 100)
    ).with_columns(pl.col("sales").cast(pl.Float32))

    current_df = history
    forecasts = []

    logger.info("Starting recursive loop (28 days)...")
    
    # We need a meta map to append rows properly
    meta_cols = ["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
    meta_map = history.select(meta_cols).unique(subset=["id"])
    
    for day in range(1, horizon + 1):
        target_d = f"d_{last_day + day}"
        print(f"Predicting {target_d}...", end="\\r")

        # 1. Add placeholder row for target_d (sales=0) so prepare_features can process it
        # We must add it for all 30,490 items
        new_rows = pl.DataFrame(
            {
                "id": sorted(unique_ids),
                "d": [target_d] * items_count,
                "sales": [0.0] * items_count,
            },
            schema={"id": pl.String, "d": pl.String, "sales": pl.Float32}
        ).join(meta_map, on="id", how="left")
        
        # Concat the new rows
        current_df = pl.concat([current_df, new_rows.select(current_df.columns)], how="vertical")
        
        # 2. Feature Engineering (Recalculate lags for the target day)
        rich_df = prepare_features(current_df, cal, prices)

        # 3. Filter to only the day we are predicting
        day_df = rich_df.filter(pl.col("d") == target_d)
        
        # Prepare features for LightGBM
        features = [c for c in day_df.columns if c not in ["sales", "date", "id", "d", "wm_yr_wk", "weekday"]]
        X_val = day_df.select(features).to_pandas()
        X_val = X_val[model_features] # Reorder
        
        # Cast categoricals
        cat_cols = X_val.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X_val[col] = X_val[col].astype("category")

        # 4. Predict
        preds = model.predict(X_val)
        preds = np.maximum(preds, 0) # No negative sales
        
        forecasts.append(preds)
        
        # 5. Update the current_df with the actual predictions so the next day's lag uses them
        # We need to update the last `items_count` rows of current_df
        # Since we just appended them, they are at the end, but to be safe, we join or update
        # Actually, since Polars dataframes are immutable, we recreate current_df
        # Replacing the 0.0 sales with preds for the target_d
        
        # We can just drop the placeholder rows and append the predicted rows!
        current_df = current_df.filter(pl.col("d") != target_d)
        
        pred_rows = pl.DataFrame(
            {
                "id": day_df["id"].to_list(),
                "d": [target_d] * items_count,
                "sales": preds.tolist(),
            },
            schema={"id": pl.String, "d": pl.String, "sales": pl.Float32}
        ).join(meta_map, on="id", how="left")
        
        current_df = pl.concat([current_df, pred_rows.select(current_df.columns)], how="vertical")
        
    logger.info("\\nRecursive Inference Complete!")
    
    # 6. Format Output
    forecast_array = np.array(forecasts).T
    out_df = pd.DataFrame(forecast_array, columns=[f"F{i}" for i in range(1, 29)])
    out_df["id"] = day_df["id"].to_list() # Uses the sorted IDs from day_df

    # Save
    out_path = paths.FORECASTS_DIR / "forecast_lgbm.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved LightGBM forecast to {out_path}")

if __name__ == "__main__":
    main()
