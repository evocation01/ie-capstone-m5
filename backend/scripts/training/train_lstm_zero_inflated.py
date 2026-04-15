import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl_lightning
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("train_lstm_zero_inflated")
warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=90, pred_length=28):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Group by item_id
        self.groups = data.groupby('id')
        self.item_ids = list(self.groups.groups.keys())

        # Pre-compute sequences for each item
        self.sequences = []
        for item_id in self.item_ids:
            item_data = self.groups.get_group(item_id)['sales'].values
            if len(item_data) >= seq_length + pred_length:
                # Use log transform to handle zeros: log(x + 1) maps 0 -> 0, positive -> positive
                item_data_transformed = torch.log(torch.tensor(item_data, dtype=torch.float32) + 1)

                for i in range(len(item_data) - seq_length - pred_length + 1):
                    seq = item_data_transformed[i:i+seq_length]
                    target = item_data_transformed[i+seq_length:i+seq_length+pred_length]
                    self.sequences.append((seq, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return seq, target


print("Dataset class defined")


class LSTMForecaster(pl_lightning.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=28, dropout=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len)
        lstm_out, _ = self.lstm(x.unsqueeze(-1))  # Add feature dimension: (batch, seq_len, 1)
        last_hidden = lstm_out[:, -1, :]  # Take last time step: (batch, hidden_size)
        output = self.fc(last_hidden)  # (batch, output_size)
        return output  # Return log-transformed predictions

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, seq_len), y: (batch, pred_len)
        y_hat = self(x)  # (batch, pred_len)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

print("LSTM model class defined")


def main():
    logger.info("🚀 Starting Custom LSTM Training for Zero-Inflated Data...")

    # 1. Load Data
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return

    logger.info("Loading data with Polars...")
    # Filter to CA_1 for initial testing
    df_pl = pl.read_parquet(data_path).filter(pl.col("store_id") == "CA_1")

    logger.info("Cleaning data...")
    df_pl = df_pl.fill_null(0).fill_nan(0)

    logger.info("Converting to Pandas...")
    df = df_pl.to_pandas()

    # Filter out items with zero total sales
    logger.info("Filtering out inactive items...")
    total_sales = df.groupby("id")["sales"].sum()
    active_ids = total_sales[total_sales > 0].index
    df = df[df["id"].isin(active_ids)].copy()
    logger.info(f"Kept {len(active_ids)} active items.")

    # Sort by date and id
    df = df.sort_values(['id', 'date']).copy()

    # Create time index
    df['time_idx'] = df.groupby('id').cumcount()

    # Split train/validation - use shorter sequences for feasibility
    max_time = df['time_idx'].max()
    train_cutoff = max_time - 28  # Last 28 days for validation

    train_data = df[df['time_idx'] <= train_cutoff]
    val_data = df[df['time_idx'] > train_cutoff]

    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # 2. Create Datasets with shorter sequences
    seq_length = 14  # Much shorter for validation feasibility
    pred_length = 7   # Shorter prediction horizon
    train_dataset = TimeSeriesDataset(train_data, seq_length=seq_length, pred_length=pred_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length=seq_length, pred_length=pred_length)

    logger.info(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. Initialize Model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=pred_length,  # Match pred_length
        dropout=0.1,
        learning_rate=1e-3
    )

    # 4. Train
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(paths.MODELS_DIR),
        filename="lstm_zero_inflated_best",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )

    trainer = pl_lightning.Trainer(
        max_epochs=10,  # Start with fewer epochs for testing
        accelerator="cpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, early_stop_callback],
        limit_train_batches=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info(f"Training Complete. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()