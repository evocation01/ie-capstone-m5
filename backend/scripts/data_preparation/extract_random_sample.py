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
    logger.info("🚀 Extracting Stratified Sample Forecasts (10 items)...")

    forecast_path = paths.FORECASTS_DIR / "forecast_lgbm.csv"
    if not forecast_path.exists():
        logger.error(f"Source forecast file not found at {forecast_path}")
        return

    # Load Forecasts
    df = pd.read_csv(forecast_path)
    logger.info(f"Loaded {len(df)} item forecasts.")

    # Stratified Sampling logic
    # ID format: DEPT_CAT_ID_STORE_validation (e.g. FOODS_1_001_CA_1_validation)
    # We want to extract DEPT_CAT (e.g. FOODS_1)

    # Create a temporary column for grouping
    # We take the first two parts of the split string (e.g., "FOODS", "1") and join them
    df["group"] = df["id"].apply(lambda x: "_".join(x.split("_")[:2]))

    groups = df["group"].unique()
    logger.info(f"Found groups: {groups}")

    sample_ids = []

    # We want ~10 items. There are usually 7 groups (FOODS_1,2,3, HOBBIES_1,2, HOUSEHOLD_1,2).
    # We will pick 1 from each group first to ensure diverse coverage.

    random.seed(42)

    for g in groups:
        group_df = df[df["group"] == g]
        if not group_df.empty:
            chosen = random.choice(group_df["id"].tolist())
            sample_ids.append(chosen)

    # Now we have ~7 items (one per group). We need a few more to make 10.
    target_size = 10
    current_count = len(sample_ids)
    needed = target_size - current_count

    if needed > 0:
        remaining_df = df[~df["id"].isin(sample_ids)]
        if len(remaining_df) >= needed:
            extras = random.sample(remaining_df["id"].tolist(), needed)
            sample_ids.extend(extras)
        else:
            sample_ids.extend(remaining_df["id"].tolist())

    logger.info(f"Selected {len(sample_ids)} Stratified Sample IDs.")

    # Filter for the sampled IDs and drop helper col
    sample_df = df[df["id"].isin(sample_ids)].drop(columns=["group"])

    # Save the sample to a new file
    out_path = paths.FORECASTS_DIR / "forecast_lgbm_sample10.csv"
    sample_df.to_csv(out_path, index=False)

    logger.info(f"✅ Saved stratified sample forecast to {out_path}")


if __name__ == "__main__":
    main()
