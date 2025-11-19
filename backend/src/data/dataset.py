import numpy as np
import polars as pl
import torch
from src.utils.logger import get_logger
from torch.utils.data import Dataset

logger = get_logger(__name__)


class M5Dataset(Dataset):
    def __init__(self, data: pl.DataFrame, seq_len: int = 90):
        """
        Custom PyTorch Dataset for M5 Forecasting.

        Args:
            data (pl.DataFrame): The feature-rich dataframe.
            seq_len (int): Lookback window size (how many past days to feed the LSTM).
        """
        self.seq_len = seq_len

        # 1. Pre-process Data for fast indexing
        # We need to group data by time series ID (item_id + store_id)
        # To make this fast, we sort by ID and Date, then convert to numpy arrays.
        # Pandas/Numpy indexing is faster than Polars for random access in __getitem__

        logger.info("Converting Polars to Numpy for Dataset (this takes RAM)...")

        # Select only necessary columns
        # We assume features are already numerical/categorical encoded
        # Features: Lags, Price, Calendar
        self.feature_cols = [
            c for c in data.columns if c not in ["sales", "date", "id"]
        ]
        self.target_col = "sales"

        # Convert to numpy for speed
        # Note: This puts the WHOLE dataset in RAM as numpy arrays.
        # If 30GB is tight, we might need a more complex memory-mapped approach later.
        self.X = data.select(self.feature_cols).to_numpy().astype(np.float32)
        self.y = data.select(self.target_col).to_numpy().astype(np.float32)

        # We need to know where each time series starts and ends
        # Since we sorted by ID + Date in make_features.py, we can find indices easily
        # Let's get the count of days per series
        # Assuming all series have same length (1913 days)
        series_ids = data.select("id").to_series()
        unique_ids = series_ids.unique()
        self.num_series = len(unique_ids)
        self.series_len = len(data) // self.num_series

        logger.info(
            f"Dataset Ready. Series: {self.num_series}, Length per series: {self.series_len}"
        )

    def __len__(self):
        # How many valid sequences can we create?
        # For each series, we can start a sequence from index 'seq_len' up to the end
        valid_starts_per_series = self.series_len - self.seq_len
        return valid_starts_per_series * self.num_series

    def __getitem__(self, idx):
        """
        Returns a tuple (x_seq, y_target)
        idx is a flat index from 0 to len(dataset).
        We need to map this flat index to a specific (series_id, time_step).
        """
        # Map flat index to series_idx and time_idx
        # How many valid sequences are in one series?
        valid_len = self.series_len - self.seq_len

        series_idx = idx // valid_len
        time_offset = idx % valid_len

        # Calculate the absolute start row in the big X matrix
        # Start of this series + offset
        start_row = (series_idx * self.series_len) + time_offset
        end_row = start_row + self.seq_len

        # Get the sequence
        x_seq = self.X[start_row:end_row]  # Shape: (seq_len, num_features)

        # Target is the value at the END of the sequence (predicting t using t-seq_len ... t-1)
        # Or predicting t+1? Let's predict the NEXT value after the sequence.
        target_row = end_row

        # Safety check (should be covered by __len__ but good for debugging)
        if target_row >= len(self.y):
            target_row = len(self.y) - 1

        y_target = self.y[target_row]  # Shape: (1,)

        return torch.tensor(x_seq), torch.tensor(y_target)
