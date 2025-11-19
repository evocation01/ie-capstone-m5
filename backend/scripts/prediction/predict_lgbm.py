import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import polars as pl

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("predict_lgbm")


def main():
    logger.info("🚀 Starting LightGBM Inference Pipeline...")

    # 1. Load Model
    model_path = paths.MODELS_DIR / "baseline_lgbm.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 2. Load Data (Validation Period)
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    logger.info(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path)

    # Filter for Validation period
    split_date = "2016-03-27"
    val = df.filter(pl.col("date") > pl.lit(split_date).str.to_date())

    logger.info(f"Validation data shape: {val.shape}")

    # 3. Prepare Features
    # FIX: We MUST include item_id and store_id in the features passed to the model!
    # We only drop target ('sales'), time index ('date'), and unique row identifier ('id')
    # Everything else (including item_id, store_id) is a valid feature.
    features = [c for c in val.columns if c not in ["sales", "date", "id"]]

    # Convert to Pandas
    X_val = val.select(features).to_pandas()

    # FIX: Robust Categorical Casting & Ordering
    # Get exact feature names from the model to ensure 100% match
    model_features = model.feature_name()

    # Check if we are missing any columns
    missing_cols = set(model_features) - set(X_val.columns)
    if missing_cols:
        logger.error(f"Missing features in validation data: {missing_cols}")
        return

    # Reorder columns to match model's expectation exactly
    X_val = X_val[model_features]

    # Identify object columns that need casting
    # In Polars they might be Int/Cat, but in Pandas conversion they might revert or need explicit 'category' dtype
    # We check all columns that ARE categorical in the original training logic
    # or just blindly cast all object columns.
    cat_cols = X_val.select_dtypes(include=["object"]).columns

    if len(cat_cols) > 0:
        logger.info(f"Casting categorical columns: {list(cat_cols)}")
        for col in cat_cols:
            X_val[col] = X_val[col].astype("category")

    # 4. Predict
    logger.info("Generating predictions...")

    try:
        preds = model.predict(X_val)
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        raise e

    # 5. Format Output
    output_df = val.select(["id", "date"]).to_pandas()
    output_df["pred"] = preds

    # Pivot
    pivot_df = output_df.pivot(index="id", columns="date", values="pred")
    pivot_df.columns = [f"F{i+1}" for i in range(28)]
    pivot_df = pivot_df.reset_index()

    # Save
    out_path = paths.EXPERIMENTS_DIR / "forecast_lgbm.csv"
    pivot_df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved LightGBM forecast to {out_path}")


if __name__ == "__main__":
    main()
