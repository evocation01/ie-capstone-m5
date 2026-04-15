import sys
import argparse
from pathlib import Path

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error
from src.config import paths
from src.utils.logger import get_logger
import lightgbm as lgb

logger = get_logger("validate_multi_store")


def validate_store(store_id: str):
    """Validate LightGBM performance on a specific store"""
    logger.info(f"🔍 Validating LightGBM on store: {store_id}")

    # 1. Load Feature-Rich Data
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error("Data not found! Run scripts/make_features.py first.")
        return None

    logger.info(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path).filter(pl.col("store_id") == store_id)

    if df.height == 0:
        logger.error(f"No data found for store {store_id}")
        return None

    logger.info(f"Store {store_id} has {df.height} rows")

    # 2. Time-Based Split (same as original)
    split_date = "2016-03-27"
    logger.info(f"Splitting data at {split_date}...")
    train = df.filter(pl.col("date") <= pl.lit(split_date).str.to_date())
    val = df.filter(pl.col("date") > pl.lit(split_date).str.to_date())

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Val shape:   {val.shape}")

    if val.height == 0:
        logger.error("Validation set is empty!")
        return None

    # 3. Prepare X and y
    target = "sales"
    features = [c for c in train.columns if c not in ["sales", "date", "id", "store_id", "state_id"]]

    logger.info(f"Features ({len(features)}): {features[:5]}...")  # Show first 5

    # Convert to Pandas and handle categoricals
    X_train = train.select(features).to_pandas()
    y_train = train.select(target).to_pandas().values.ravel()
    X_val = val.select(features).to_pandas()
    y_val = val.select(target).to_pandas().values.ravel()

    # Handle categorical features for LightGBM
    categorical_features = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            # Convert to category
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            categorical_features.append(col)

    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # 4. Train LightGBM with same hyperparameters
    logger.info("Training LightGBM...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # 5. Evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    logger.info(f"Store {store_id} - RMSE: {rmse:.4f}")

    # 6. Calculate financial impact (simplified)
    # Using same cost structure as before
    holding_cost_per_unit = 1.0
    stockout_cost_per_unit = 10.0
    service_level = 0.95
    z_score = 1.645  # 95% service level
    lead_time = 1

    # Safety stock per item (simplified - using average RMSE per store)
    safety_stock = z_score * rmse * np.sqrt(lead_time)

    # Simplified cost calculation (would need per-item calculation for accuracy)
    avg_sales = np.mean(y_val)
    holding_cost = safety_stock * holding_cost_per_unit
    # Estimate stockouts (simplified)
    stockout_rate = max(0, 1 - service_level)
    stockout_cost = stockout_rate * avg_sales * len(y_val) * stockout_cost_per_unit
    total_cost = holding_cost + stockout_cost

    results = {
        'store_id': store_id,
        'rmse': rmse,
        'safety_stock': safety_stock,
        'holding_cost': holding_cost,
        'stockout_cost': stockout_cost,
        'total_cost': total_cost,
        'avg_sales': avg_sales,
        'sample_count': len(y_val)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate LightGBM on multiple stores')
    parser.add_argument('--stores', nargs='+', default=['CA_1', 'CA_2', 'CA_3'],
                        help='List of store IDs to validate')
    args = parser.parse_args()

    logger.info(f"🚀 Starting Multi-Store Validation for stores: {args.stores}")

    results = []
    for store_id in args.stores:
        result = validate_store(store_id)
        if result:
            results.append(result)

    # Summary
    if results:
        logger.info("\n📊 Multi-Store Validation Summary:")
        logger.info("-" * 80)
        for result in results:
            logger.info(f"Store {result['store_id']:>6}: RMSE={result['rmse']:.3f}, "
                       f"Total Cost=${result['total_cost']:,.0f}, "
                       f"Samples={result['sample_count']}")

        # Check consistency
        rmses = [r['rmse'] for r in results]
        rmse_std = np.std(rmses)
        logger.info(f"\nConsistency Check - RMSE Std Dev: {rmse_std:.4f}")
        if rmse_std < 0.2:
            logger.info("✅ Model performance is consistent across stores")
        else:
            logger.info("⚠️  Model performance varies across stores")


if __name__ == "__main__":
    main()