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
    logger.info("🚀 Extracting Sample Forecasts for analysis...")

    forecast_path = paths.FORECASTS_DIR / "forecast_lgbm.csv"
    if not forecast_path.exists():
        logger.error(f"Source forecast file not found at {forecast_path}")
        return

    # Load Forecasts
    df = pd.read_csv(forecast_path)
    logger.info(f"Loaded {len(df)} item forecasts.")

    # Pick 100 Random Items for the sample
    # We fix the seed so the sample is reproducible
    random.seed(42)
    if len(df) < 100:
        logger.warning("Fewer than 100 items in forecast, using all items for sample.")
        sample_ids = df["id"].tolist()
    else:
        sample_ids = random.sample(df["id"].tolist(), 100)

    logger.info(f"Selected {len(sample_ids)} Sample IDs for the subset.")

    # Filter for the sampled IDs
    sample_df = df[df["id"].isin(sample_ids)]

    # Save the sample to a new file
    out_path = paths.FORECASTS_DIR / "forecast_lgbm_sample100.csv"
    sample_df.to_csv(out_path, index=False)

    logger.info(f"✅ Saved sample forecast to {out_path}")


if __name__ == "__main__":
    main()
