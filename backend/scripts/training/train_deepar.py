import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl_lightning
import pandas as pd
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import RobustScaler

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("train_deepar")
warnings.filterwarnings("ignore")


def main():
    logger.info("🚀 Starting DeepAR Training Pipeline...")

    # 1. Load Data
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return

    logger.info("Loading data with Polars...")
    # Filter to CA_1 to save RAM initially
    df_pl = pl.read_parquet(data_path).filter(pl.col("store_id") == "CA_1")

    logger.info("Cleaning data (filling nulls)...")
    df_pl = df_pl.fill_null(0).fill_nan(0)

    logger.info("Converting to Pandas for PyTorch Forecasting...")
    df = df_pl.to_pandas()

    # Ensure target 'sales' is FLOAT
    df["sales"] = df["sales"].astype(float)

    # FIX: Filter out items with ZERO sales history.
    # DeepAR breaks if a time series is completely flat/zero because Normalizer divides by 0 variance/mean.
    # We calculate total sales per ID and keep only those > 0.
    logger.info("Filtering out 'dead' items (0 total sales)...")
    total_sales = df.groupby("id")["sales"].sum()
    active_ids = total_sales[total_sales > 0].index
    df = df[df["id"].isin(active_ids)].copy()
    logger.info(f"Dropped {len(total_sales) - len(active_ids)} inactive items.")

    # Ensure Categoricals are Strings
    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "unknown")

    # Create integer time index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        df["time_idx"] = (df["date"] - min_date).dt.days
    else:
        df["time_idx"] = df.groupby("id").cumcount()

    # 2. Create TimeSeriesDataSet
    max_prediction_length = 28
    max_encoder_length = 90
    training_cutoff = df["time_idx"].max() - max_prediction_length

    logger.info("Creating TimeSeriesDataSet...")

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["item_id", "dept_id", "cat_id"],
        time_varying_known_categoricals=["event_name_1", "event_type_1"],
        time_varying_known_reals=[
            "sell_price",
            "price_momentum",
            "is_weekend",
        ],
        time_varying_unknown_reals=["sales"],
        scalers={
            "sell_price": RobustScaler(),
            "price_momentum": RobustScaler(),
        },
        # center=False is critical for NegativeBinomial
        target_normalizer=GroupNormalizer(
            groups=["id"], transformation="softplus", center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    batch_size = 64
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    # 3. Initialize Model
    logger.info("Initializing DeepAR Model...")

    net = DeepAR.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        rnn_layers=2,
        dropout=0.1,
    )

    # 4. Train
    logger.info("Starting Training...")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(paths.MODELS_DIR),
        filename="deepar_best",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )

    # CRITICAL FIX: Force CPU.
    # MPS (Mac GPU) often produces NaNs with LSTMs/NegativeBinomial math.
    # If this runs successfully, the issue was MPS instability.
    accelerator = "cpu"
    logger.info(f"Using accelerator: {accelerator} (Forced for stability)")

    trainer = pl_lightning.Trainer(
        max_epochs=15,
        accelerator=accelerator,
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, early_stop_callback],
        limit_train_batches=1.0,
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info(
        f"Training Complete. Best model saved at: {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
