import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl_lightning
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger
from scripts.training.train_lstm_zero_inflated import LSTMForecaster, TimeSeriesDataset

logger = get_logger("validate_lstm_results")
warnings.filterwarnings("ignore")


def validate_lstm_model():
    """Validate the trained LSTM model on test data"""
    logger.info("🔬 Starting LSTM Model Validation...")

    # 1. Load Data (same as training)
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return None

    logger.info("Loading data...")
    df_pl = pl.read_parquet(data_path).filter(pl.col("store_id") == "CA_1")
    df_pl = df_pl.fill_null(0).fill_nan(0)

    logger.info("Converting to Pandas...")
    df = df_pl.to_pandas()
    total_sales = df.groupby("id")["sales"].sum()
    active_ids = total_sales[total_sales > 0].index
    df = df[df["id"].isin(active_ids)].copy()

    df = df.sort_values(['id', 'date']).copy()
    df['time_idx'] = df.groupby('id').cumcount()

    # Split same as training
    max_time = df['time_idx'].max()
    train_cutoff = max_time - 28
    train_data = df[df['time_idx'] <= train_cutoff]
    val_data = df[df['time_idx'] > train_cutoff]

    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # 2. Create model and load weights manually (compatibility issue with PL versions)
    logger.info("Creating model with same architecture...")
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=7,  # pred_length
        dropout=0.1,
        learning_rate=1e-3
    )

    # Try to load weights if available
    model_path = paths.MODELS_DIR / "lstm_best.pt"
    if model_path.exists():
        logger.info(f"Loading weights from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Could not load weights: {e}. Using randomly initialized model.")

    model.eval()

    # 3. Create validation dataset (same parameters as training)
    seq_length = 14
    pred_length = 7
    val_dataset = TimeSeriesDataset(val_data, seq_length=seq_length, pred_length=pred_length)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    logger.info(f"Validation sequences: {len(val_dataset)}")

    # 4. Make predictions
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            pred = model(x)

            # Convert back from log scale to original scale
            pred_original = torch.exp(pred) - 1  # inverse of log(x + 1)
            target_original = torch.exp(y) - 1

            all_predictions.extend(pred_original.flatten().cpu().numpy())
            all_targets.extend(target_original.flatten().cpu().numpy())

    # 5. Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = np.mean(np.abs(np.array(all_targets) - np.array(all_predictions)))

    logger.info("Validation Results:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Number of predictions: {len(all_predictions)}")

    # 6. Compare with baseline models
    # Naive forecast (last value repeated)
    naive_predictions = []
    for item_id in val_data['id'].unique():
        item_data = val_data[val_data['id'] == item_id]
        last_train_value = train_data[train_data['id'] == item_id]['sales'].iloc[-1] if len(train_data[train_data['id'] == item_id]) > 0 else 0
        naive_predictions.extend([last_train_value] * len(item_data))

    naive_rmse = np.sqrt(mean_squared_error(val_data['sales'].values[:len(naive_predictions)], naive_predictions))
    logger.info(f"Naive RMSE: {naive_rmse:.4f}")

    # 7. Calculate financial impact
    results = {
        'rmse': rmse,
        'mae': mae,
        'naive_rmse': naive_rmse,
        'predictions': all_predictions[:10],  # Sample predictions
        'targets': all_targets[:10],  # Sample targets
        'improvement_vs_naive': (naive_rmse - rmse) / naive_rmse * 100
    }

    return results


if __name__ == "__main__":
    results = validate_lstm_model()
    if results:
        print("\n" + "="*50)
        print("LSTM VALIDATION RESULTS")
        print("="*50)
        print(".4f")
        print(".4f")
        print(".4f")
        print(".1f")
        print("="*50)