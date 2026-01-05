from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModel:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with early stopping.
        """
        logger.info("Preparing LightGBM datasets...")
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        logger.info("Starting training...")
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )
        logger.info("Training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: Path):
        logger.info(f"Saving model to {path}")
        joblib.dump(self.model, path)

    def load(self, path: Path):
        logger.info(f"Loading model from {path}")
        self.model = joblib.load(path)
