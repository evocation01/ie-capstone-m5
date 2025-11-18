from pathlib import Path

# Calculate the root of the backend project
# Assumes this file is at: backend/src/config/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model Directories
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Verify directories exist
if not RAW_DATA_DIR.exists():
    print(f"WARNING: Raw data directory not found at {RAW_DATA_DIR}")
