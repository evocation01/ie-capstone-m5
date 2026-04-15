import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("train_tweedie")


def main():
    logger.info("🚀 Starting LightGBM Tweedie Training (for zero-inflated data)...")

    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return

    # Load CA_1 store only
    logger.info("Loading CA_1 store data...")
    df = pl.read_parquet(data_path).filter(pl.col("store_id") == "CA_1")
    df = df.fill_null(0).fill_nan(0)
    logger.info(f"Loaded {df.height} rows")

    # Time-Based Split (last 28 days for validation)
    split_idx = df.height - 28 * 3049  # ~3049 items per day
    train = df.slice(0, split_idx)
    val = df.slice(split_idx)

    logger.info(f"Train: {train.height}, Val: {val.height}")

    # Features and target
    features = [c for c in train.columns if c not in ["sales", "date", "id"]]
    logger.info(f"Features: {len(features)}")

    X_train = train.select(features).to_pandas()
    y_train = train.select("sales").to_pandas().values.ravel()

    X_val = val.select(features).to_pandas()
    y_val = val.select("sales").to_pandas().values.ravel()

    # Cast object columns to category
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    # Tweedie params - designed for zero-inflated count data!
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,  # 1=Poisson, 2=Gamma - 1.5 is good for retail
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    logger.info("Training LightGBM with Tweedie loss...")
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    logger.info(f"✅ Validation RMSE (Tweedie): {rmse:.4f}")

    # Compare: What's the zero rate?
    zero_rate = (y_val == 0).mean()
    logger.info(f"Zero rate in validation: {zero_rate:.1%}")

    # Save
    model.save_model(str(paths.MODELS_DIR / "lgbm_tweedie.txt"))
    logger.info(f"Model saved to {paths.MODELS_DIR / 'lgbm_tweedie.txt'}")


if __name__ == "__main__":
    main()