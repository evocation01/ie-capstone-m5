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

logger = get_logger("tweedie_validation")


def main():
    logger.info("=== Running Tweedie on Original Validation ===")

    # Load same raw data as optimize.py
    raw = pd.read_csv(paths.RAW_DATA_DIR / "sales_train_validation.csv")
    
    # Get CA_1 items (same as optimize.py)
    ca1 = raw[raw["store_id"] == "CA_1"]
    ca1_items = ca1["item_id"].unique()  # All items
    
    # Same columns as optimize.py
    all_d = [c for c in raw.columns if c.startswith("d_")]
    val_cols = all_d[-28:]  # Last 28 days
    naive_cols = all_d[-56:-28]  # 28 days before validation
    
    # Load Tweedie model
    model = lgb.Booster(model_file=str(paths.MODELS_DIR / "lgbm_tweedie.txt"))
    
    # Load processed data using features
    df = pl.read_parquet(paths.PROCESSED_DATA_DIR / "final_train.parquet").filter(pl.col("store_id") == "CA_1")
    df = df.fill_null(0)
    
    # Get last 28 days (same split as Tweedie training)
    split_idx = df.height - 28 * 3049
    val = df.slice(split_idx)
    
    features = [c for c in val.columns if c not in ["sales", "date", "id"]]
    X_val = val.select(features).to_pandas()
    y_val_actual = val.select("sales").to_pandas().values.ravel()

    # Handle categories
    for col in X_val.select_dtypes(include=["object"]).columns:
        X_val[col] = X_val[col].astype("category")

    # Predict
    tweedie_preds = model.predict(X_val)

    # Reshape and aggregate by item
    n_items = 3049
    tweedie_by_item = []
    actual_by_item = []
    
    for i in range(n_items):
        start = i * 28
        end = start + 28
        tweedie_by_item.append(tweedie_preds[start:end].mean())
        actual_by_item.append(y_val_actual[start:end].mean())

    tweedie_by_item = np.array(tweedie_by_item)
    actual_by_item = np.array(actual_by_item)

    # Calculate RMSE
    rmse_tweedie = np.sqrt(np.mean((tweedie_by_item - actual_by_item) ** 2))
    
    # Naive RMSE
    naive_by_item = []
    for i in range(n_items):
        start = i * 28
        end = start + 28
        hist = y_val_actual[start:end]
        naive_by_item.append(hist.mean())
    naive_by_item = np.array(naive_by_item)
    rmse_naive = np.sqrt(np.mean((naive_by_item - actual_by_item) ** 2))

    logger.info(f"Naive RMSE: {rmse_naive:.4f}")
    logger.info(f"Tweedie RMSE: {rmse_tweedie:.4f}")

    # Calculate Costs (same as optimize.py)
    HOLDING = 1.0
    STOCKOUT = 10.0

    tweedie_matrix = np.tile(tweedie_by_item, (28, 1)).T
    naive_matrix = np.tile(naive_by_item, (28, 1)).T
    actual_matrix = np.tile(actual_by_item, (28, 1)).T

    # Tweedie costs
    diff = tweedie_matrix - actual_matrix
    holding_t = np.sum(np.maximum(diff, 0)) * HOLDING
    stockout_t = np.sum(np.maximum(-diff, 0)) * STOCKOUT
    total_tweedie = holding_t + stockout_t

    # Naive costs
    diff_n = naive_matrix - actual_matrix
    holding_n = np.sum(np.maximum(diff_n, 0)) * HOLDING
    stockout_n = np.sum(np.maximum(-diff_n, 0)) * STOCKOUT
    total_naive = holding_n + stockout_n

    logger.info(f"Naive Total Cost: ${total_naive:,.0f}")
    logger.info(f"Tweedie Total Cost: ${total_tweedie:,.0f}")

    # Compare with original LightGBM
    orig_lgbm_total = 515513
    orig_naive_total = 640703

    logger.info("")
    logger.info("=== COMPARISON WITH ORIGINAL ===")
    logger.info(f"Original Naive: ${orig_naive_total:,}")
    logger.info(f"Original LightGBM: ${orig_lgbm_total:,} (19.5% vs Naive)")
    logger.info(f"Now Naive: ${total_naive:,}")
    logger.info(f"Now Tweedie: ${total_tweedie:,} ({((total_naive - total_tweedie) / total_naive * 100):.1f}% vs Naive)")

    # Save to JSON
    results = pd.DataFrame([{
        "Model": "Tweedie",
        "RMSE": rmse_tweedie,
        "Total_Cost": total_tweedie,
        "Savings_vs_Naive": ((total_naive - total_tweedie) / total_naive * 100)
    }])
    results.to_csv(paths.RESULTS_DIR / "tweedie_comparison.csv", index=False)
    logger.info(f"Saved to {paths.RESULTS_DIR / 'tweedie_comparison.csv'}")


if __name__ == "__main__":
    main()