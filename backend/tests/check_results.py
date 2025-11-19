import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.logger import get_logger

logger = get_logger("check_results")


def check_forecast_file(name, path):
    if not path.exists():
        logger.error(f"❌ MISSING: {name} at {path}")
        return False

    try:
        df = pd.read_csv(path)

        # 1. Check Shape
        if df.empty:
            logger.error(f"❌ EMPTY: {name} contains no rows!")
            return False

        # 2. Check Columns
        expected_cols = ["id"] + [f"F{i}" for i in range(1, 29)]
        if not all(col in df.columns for col in expected_cols):
            logger.error(f"❌ BAD SCHEMA: {name} is missing F1..F28 or id columns.")
            return False

        # 3. Check Values (Sparsity & Negatives)
        f_data = df.iloc[:, :-1]  # Exclude ID
        min_val = f_data.min().min()
        max_val = f_data.max().max()
        zero_count = (f_data == 0).sum().sum()
        total_cells = f_data.size
        sparsity = (zero_count / total_cells) * 100

        if min_val < 0:
            logger.error(f"❌ INVALID VALUES: {name} contains negatives ({min_val})!")
            return False

        logger.info(f"✅ {name} is valid.")
        logger.info(f"   - Shape: {df.shape}")
        logger.info(f"   - Range: [{min_val:.4f}, {max_val:.4f}]")
        logger.info(f"   - Sparsity: {sparsity:.2f}% Zeros")

        return True

    except Exception as e:
        logger.error(f"❌ CRASH: Could not read {name}. Error: {e}")
        return False


def check_final_results(path):
    if not path.exists():
        logger.error(f"❌ MISSING: Final Results at {path}")
        return False

    df = pd.read_csv(path)
    required = ["Model", "RMSE", "Total_Cost"]
    if not all(col in df.columns for col in required):
        logger.error("❌ BAD SCHEMA: Final Results missing columns.")
        return False

    logger.info("✅ Final Results Summary:")
    print(df.to_string(index=False))
    return True


def main():
    logger.info("🔍 Running Sanity Checks on Experiment Outputs...")

    all_good = True

    # Check Forecasts
    if not check_forecast_file(
        "LSTM Forecast", paths.EXPERIMENTS_DIR / "forecast_lstm.csv"
    ):
        all_good = False

    if not check_forecast_file(
        "LightGBM Forecast", paths.EXPERIMENTS_DIR / "forecast_lgbm.csv"
    ):
        all_good = False

    # Check Optimization Results
    if not check_final_results(paths.EXPERIMENTS_DIR / "final_results.csv"):
        all_good = False

    if all_good:
        logger.info("\n🎉 ALL CHECKS PASSED! Your results are ready for the report.")
    else:
        logger.error("\n⚠️ SOME CHECKS FAILED. Do not send these files yet.")


if __name__ == "__main__":
    main()
