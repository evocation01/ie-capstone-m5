import random
import sys
from pathlib import Path

import pandas as pd

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("extract_sample")


def main():
    logger.info("🚀 Extracting Sample Forecasts for Team...")

    forecast_path = paths.EXPERIMENTS_DIR / "forecast_lgbm.csv"
    if not forecast_path.exists():
        logger.error(f"Forecast file not found at {forecast_path}")
        return

    # Load Forecasts
    df = pd.read_csv(forecast_path)
    logger.info(f"Loaded {len(df)} item forecasts.")

    # Pick 10 Random Items
    # We fix the seed so you send the SAME 5 items if you run this again
    random.seed(42)
    sample_ids = random.sample(df["id"].tolist(), 100)

    logger.info(f"Selected Sample IDs: {sample_ids}")

    # Filter
    sample_df = df[df["id"].isin(sample_ids)]

    # Save
    out_path = paths.EXPERIMENTS_DIR / "forecast_lgbm_sample50.csv"
    sample_df.to_csv(out_path, index=False)

    logger.info(f"✅ Saved sample to {out_path}")
    logger.info("Send this file to your Optimization Lead!")


if __name__ == "__main__":
    main()
