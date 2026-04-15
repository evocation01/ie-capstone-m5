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

logger = get_logger("tweedie_compare")


def main():
    logger.info("=== Tweedie vs MSE LightGBM Comparison ===")

    # Load processed data
    df = pl.read_parquet(paths.PROCESSED_DATA_DIR / "final_train.parquet").filter(pl.col("store_id") == "CA_1")
    df = df.fill_null(0)

    features = [c for c in df.columns if c not in ["sales", "date", "id"]]
    
    # Split: last 28 days validation (same setup as original)
    split_idx = df.height - 28 * 3049
    train = df.slice(0, split_idx)
    val = df.slice(split_idx)

    X_train = train.select(features).to_pandas()
    y_train = train.select("sales").to_pandas().values.ravel()
    X_val = val.select(features).to_pandas()
    y_val = val.select("sales").to_pandas().values.ravel()

    # Handle categories
    for col in X_train.select_dtypes(include=["object"]).columns:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Train MSE model (like original)
    logger.info("Training MSE LightGBM...")
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)
    
    mse_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }
    mse_model = lgb.train(mse_params, train_set, 200, valid_sets=[val_set])

    # MSE predictions
    preds_mse = mse_model.predict(X_val)
    rmse_mse = np.sqrt(mean_squared_error(y_val, preds_mse))
    logger.info(f"MSE RMSE: {rmse_mse:.4f}")

    # Load Tweedie model
    tweedie_model = lgb.Booster(model_file=str(paths.MODELS_DIR / "lgbm_tweedie.txt"))
    preds_tweedie = tweedie_model.predict(X_val)
    rmse_tweedie = np.sqrt(mean_squared_error(y_val, preds_tweedie))
    logger.info(f"Tweedie RMSE: {rmse_tweedie:.4f}")

    # Aggregate by item (3049 items, 28 days each)
    n_items = 3049
    preds_mse_by_item = preds_mse.reshape(n_items, 28).mean(axis=1)
    preds_tweedie_by_item = preds_tweedie.reshape(n_items, 28).mean(axis=1)
    actual_by_item = y_val.reshape(n_items, 28).mean(axis=1)

    # Costs
    HOLDING = 1.0
    STOCKOUT = 10.0

    # MSE costs
    mse_matrix = np.tile(preds_mse_by_item, (28, 1)).T
    actual_matrix = np.tile(actual_by_item, (28, 1)).T
    diff = mse_matrix - actual_matrix
    holding_mse = np.sum(np.maximum(diff, 0)) * HOLDING
    stockout_mse = np.sum(np.maximum(-diff, 0)) * STOCKOUT
    total_mse = holding_mse + stockout_mse

    # Tweedie costs
    tw_matrix = np.tile(preds_tweedie_by_item, (28, 1)).T
    diff_tw = tw_matrix - actual_matrix
    holding_tw = np.sum(np.maximum(diff_tw, 0)) * HOLDING
    stockout_tw = np.sum(np.maximum(-diff_tw, 0)) * STOCKOUT
    total_tw = holding_tw + stockout_tw

    logger.info("")
    logger.info("=== RESULTS (SAME VALIDATION) ===")
    logger.info(f"MSE LightGBM - RMSE: {rmse_mse:.4f}, Total: ${total_mse:,.0f}")
    logger.info(f"Tweedie LightGBM - RMSE: {rmse_tweedie:.4f}, Total: ${total_tw:,.0f}")

    # Best?
    if total_tw < total_mse:
        diff_pct = (total_mse - total_tw) / total_mse * 100
        logger.info(f"Tweedie is BETTER by {diff_pct:.1f}%")
    else:
        diff_pct = (total_tw - total_mse) / total_tw * 100
        logger.info(f"MSE is BETTER by {diff_pct:.1f}%")

    # Print for frontend
    print(f"\n=== SUMMARY ===")
    print(f"MSE RMSE: {rmse_mse:.4f}")
    print(f"Tweedie RMSE: {rmse_tweedie:.4f}")
    print(f"MSE Total: ${total_mse:,.0f}")
    print(f"Tweedie Total: ${total_tw:,.0f}")
    print(f"Baseline (Naive) used for %: use actual last 28 days mean")


if __name__ == "__main__":
    main()