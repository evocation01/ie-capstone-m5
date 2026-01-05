import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.data.dataset import M5Dataset
from src.models.lstm import M5LSTM
from src.utils.logger import get_logger

logger = get_logger("train_dl")


def main():
    logger.info("🚀 Starting Deep Learning Training Pipeline...")

    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Load Data
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    df = pl.read_parquet(data_path)

    # Filter to reduce size for dev test
    # df = df.filter(pl.col("store_id") == "CA_1")

    # --- ROBUST DATA CLEANING ---
    logger.info("Cleaning Data...")

    # 1. Clean Numerical Columns (Float/Int)
    # We fill nulls with 0, and NaNs with 0
    num_cols = [
        c
        for c, t in zip(df.columns, df.dtypes)
        if t in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]
    ]
    for col in num_cols:
        if df[col].dtype in [pl.Float32, pl.Float64]:
            df = df.with_columns(pl.col(col).fill_nan(0))
        df = df.with_columns(pl.col(col).fill_null(0))

    # --- TARGET TRANSFORMATION ---
    logger.info("Log-transforming sales target (log1p)...")
    df = df.with_columns(
        pl.when(pl.col("sales") < 0).then(0).otherwise(pl.col("sales")).alias("sales")
    )
    df = df.with_columns(pl.col("sales").log1p())

    # --- ROBUST NORMALIZATION ---
    logger.info("Normalizing numerical features...")
    float_cols = [
        c for c, t in zip(df.columns, df.dtypes) if t in [pl.Float32, pl.Float64]
    ]

    for col in float_cols:
        if col != "sales":
            mean = df.select(pl.col(col).mean()).item()
            std = df.select(pl.col(col).std()).item()

            if std is not None and std > 1e-8:
                df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
            else:
                df = df.with_columns((pl.col(col) * 0).alias(col))

    # --- ENCODING CATEGORICALS ---
    logger.info("Encoding categorical columns...")
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

    # --- FINAL NAN CHECK ---
    null_counts = df.null_count()
    for col in df.columns:
        cnt = null_counts[col][0]
        if cnt > 0:
            logger.error(f"❌ Column '{col}' still has {cnt} Nulls after cleaning!")
            df = df.with_columns(pl.col(col).fill_null(0))

    # 3. Create Dataset
    logger.info("Creating Dataset...")
    dataset = M5Dataset(df, seq_len=28)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    # 4. Initialize Model
    num_features = dataset.X.shape[1]
    model = M5LSTM(num_features=num_features).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. Training Loop with Early Stopping
    epochs = 30  # Increased max epochs
    patience = 5  # Stop if no improvement for 5 epochs
    best_val_loss = float("inf")
    early_stop_counter = 0

    logger.info(f"Starting training for {epochs} epochs (Patience: {patience})...")

    # DEBUG: Check first batch
    try:
        first_batch = next(iter(train_loader))
        inputs_dbg, targets_dbg = first_batch
        logger.info(
            f"DEBUG: Input Min: {inputs_dbg.min():.4f}, Max: {inputs_dbg.max():.4f}, Mean: {inputs_dbg.mean():.4f}"
        )
        logger.info(
            f"DEBUG: Target Min: {targets_dbg.min():.4f}, Max: {targets_dbg.max():.4f}"
        )

        if torch.isnan(inputs_dbg).any():
            logger.error(
                "❌ Inputs contain NaN in the DataLoader! Check M5Dataset numpy conversion."
            )
            return
    except Exception as e:
        logger.error(f"❌ Error fetching first batch: {e}")
        return

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets.squeeze())

            if torch.isnan(loss):
                logger.error(f"❌ Loss is NaN at Epoch {epoch}, Batch {i}!")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}", end="\r")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch+1} | Train Log-RMSE: {avg_train_loss**0.5:.4f} | Val Log-RMSE: {avg_val_loss**0.5:.4f}"
        )

        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            # Save the best model
            save_path = paths.MODELS_DIR / "lstm_best.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"   >>> New Best Model! Saved to {save_path}")
        else:
            early_stop_counter += 1
            logger.info(
                f"   ... No improvement. Patience: {early_stop_counter}/{patience}"
            )

            if early_stop_counter >= patience:
                logger.info("🛑 Early Stopping Triggered. Training finished.")
                break


if __name__ == "__main__":
    main()
