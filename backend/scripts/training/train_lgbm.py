import sys
from pathlib import Path

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error
from src.config import paths
from src.models.baseline import BaselineModel
from src.utils.logger import get_logger

logger = get_logger("train_baseline")


def main():
    logger.info("🚀 Starting Baseline Training Pipeline...")

    # 1. Load Feature-Rich Data
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error("Data not found! Run scripts/make_features.py first.")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path)

    # 2. Time-Based Split
    # Assuming 'date' column exists (Polars Date type):
    split_date = "2016-03-27"

    logger.info(f"Splitting data at {split_date}...")
    train = df.filter(pl.col("date") <= pl.lit(split_date).str.to_date())
    val = df.filter(pl.col("date") > pl.lit(split_date).str.to_date())

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Val shape:   {val.shape}")

    # Sanity check to prevent crash
    if val.height == 0:
        logger.error("Validation set is empty! Check your split date.")
        return

    # 3. Prepare X and y
    target = "sales"
    # Drop target and non-feature columns
    # We keep 'id' just for reference if needed, but exclude from features

    features = [c for c in train.columns if c not in ["sales", "date", "id"]]
    logger.info(f"Features ({len(features)}): {features}")

    # Convert to Pandas for LightGBM
    # We use to_pandas() which might convert Categoricals back to objects if not careful
    logger.info("Converting to Pandas for LightGBM...")
    X_train = train.select(features).to_pandas()
    y_train = train.select(target).to_pandas().values.ravel()

    X_val = val.select(features).to_pandas()
    y_val = val.select(target).to_pandas().values.ravel()

    # FIX: Explicitly cast object columns to 'category' for LightGBM
    # LightGBM handles 'category' dtype natively, but crashes on 'object'
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        logger.info(f"Casting object columns to category: {list(cat_cols)}")
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

    # 4. Train
    model = BaselineModel()
    model.train(X_train, y_train, X_val, y_val)

    # 5. Evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    logger.info(f"✅ Validation RMSE: {rmse:.4f}")

    # 6. Save Model
    model_path = paths.MODELS_DIR / "baseline_lgbm.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


if __name__ == "__main__":
    main()
