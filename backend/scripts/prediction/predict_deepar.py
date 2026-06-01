import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("predict_deepar")
warnings.filterwarnings("ignore")


def main():
    logger.info("🚀 Starting DeepAR Inference Pipeline...")

    # 1. Find Model Checkpoint
    # PyTorch Lightning saves checkpoints as .ckpt files.
    # We look for 'deepar_best.ckpt' or similar in models dir
    # If strict filename was used in training, use that.
    # Otherwise, find the latest .ckpt

    ckpt_files = list(paths.MODELS_DIR.glob("*.ckpt"))
    if not ckpt_files:
        logger.error(f"No .ckpt model found in {paths.MODELS_DIR}")
        return

    # Pick the most recent one or specific name
    model_path = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"Loading model from {model_path}...")

    # Load Model
    best_deepar = DeepAR.load_from_checkpoint(model_path)

    # 2. Load Data (We need history to create the prediction dataset)
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    logger.info("Loading data...")

    # Same loading logic as training
    df_pl = pl.read_parquet(data_path).filter(pl.col("store_id") == "CA_1")
    df_pl = df_pl.fill_null(0).fill_nan(0)
    df = df_pl.to_pandas()

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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        df["time_idx"] = (df["date"] - min_date).dt.days
    else:
        df["time_idx"] = df.groupby("id").cumcount()

    # 3. Create Prediction Dataset
    # We want to predict the validation period (last 28 days in df)
    # PyTorch Forecasting makes this easy: from_dataset(..., predict=True)
    # This automatically sets up the decoder to predict the "future" relative to the training cutoff

    # We need to reconstruct the training dataset object (params) to create the prediction one
    # Or better, DeepAR model stores the dataset parameters!

    # Let's re-create the dataset definition from the dataframe
    max_prediction_length = 28
    max_encoder_length = 90

    # We use the full dataframe now
    # The 'predict=True' flag tells it to predict the last 'max_prediction_length' time steps

    # Define the training dataset structure again (needed to init the validation structure)
    # Ideally we would pickle the dataset definition, but defining it again is standard
    training_cutoff = df["time_idx"].max() - max_prediction_length

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
            "time_idx",
            "sell_price",
            "price_momentum",
            "is_weekend",
        ],
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create prediction dataset
    # predict=True means: "Use the last available data as context to predict the next unknown steps"
    pred_dataset = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    pred_dataloader = pred_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    # 4. Generate Predictions
    logger.info("Generating predictions...")
    # DeepAR predicts a distribution. By default .predict() returns the mean.
    # We move model to CPU to avoid MPS errors during complex sampling if any
    best_deepar.to("cpu")

    raw_predictions = best_deepar.predict(
        pred_dataloader, mode="prediction", return_x=True
    )
    preds = raw_predictions.output  # Shape: (Batch, 28)
    x = raw_predictions.x

    # 5. Format Output
    # We need to map predictions back to IDs.
    # The dataloader shuffles or orders them? Validation DL should be deterministic.
    # But we need the IDs. TimeSeriesDataSet returns 'groups' in 'x'.

    # Get IDs from the 'x' dictionary returned by predict
    # x['decoder_target'] is not IDs.
    # x['groups'] contains the encoded group IDs. We need to decode them.

    # PyTorch Forecasting decoders
    # The dataset object has the label encoders
    ids_encoded = x["groups"]  # Shape: (Batch, 1) -> assuming 1 group 'id'
    decoder = training.categorical_encoders["id"]
    ids = decoder.inverse_transform(ids_encoded.squeeze())

    # Create DataFrame
    preds_np = preds.numpy()

    # Ensure non-negative
    preds_np = np.maximum(preds_np, 0)

    output_df = pd.DataFrame(preds_np, columns=[f"F{i+1}" for i in range(28)])
    output_df["id"] = ids

    # Save
    out_path = paths.FORECASTS_DIR / "forecast_deepar.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved DeepAR forecast to {out_path}")


if __name__ == "__main__":
    main()
