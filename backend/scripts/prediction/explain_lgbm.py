import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
import json
import logging

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("explain_lgbm")

def main():
    logger.info("🚀 Starting SHAP Explainability Analysis...")

    # 1. Load Model
    model_path = paths.MODELS_DIR / "baseline_lgbm.pkl"
    if not model_path.exists():
        logger.error("Model not found!")
        return

    logger.info(f"Loading LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    model_features = model.feature_name()

    # 2. Load Data Sample
    data_path = paths.PROCESSED_DATA_DIR / "final_train.parquet"
    logger.info(f"Loading data from {data_path}...")
    
    # We only need a small sample for SHAP to compute feature importance accurately
    # Polars is faster, but pandas is fine since we just need the tail
    df = pd.read_parquet(data_path)
    
    # We want the most recent data (the end of the dataset) to explain current model behavior
    # Sample 2000 rows randomly from the last 1,000,000 rows
    df_sample = df.tail(1000000).sample(2000, random_state=42)
    
    # Keep only the features the model uses
    X = df_sample[model_features]
    
    # Ensure categorical columns are properly cast to float via codes to bypass LightGBM pandas validation
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes
        
    # Convert entirely to float
    X = X.astype(float)

    logger.info(f"Calculating SHAP values for {len(X)} samples...")
    
    # 3. Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    # Pass numpy array to bypass pandas categorical check
    shap_values = explainer.shap_values(X.values)
    
    # shap_values might be a list if multi-class, but for regression it's a 2D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Take positive class if binary
        
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create a DataFrame for easy sorting
    importance_df = pd.DataFrame({
        "feature": model_features,
        "importance": mean_abs_shap
    })
    
    # Sort and get top 15 features
    top_features = importance_df.sort_values(by="importance", ascending=False).head(15)
    
    logger.info("Top 5 Features:")
    for _, row in top_features.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
    # 4. Export to JSON
    export_data = []
    for _, row in top_features.iterrows():
        export_data.append({
            "feature": row["feature"],
            "importance": float(row["importance"])
        })
        
    # We save it to results, and export script will pick it up
    out_dir = paths.RESULTS_DIR / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_importance.json"
    
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=4)
        
    logger.info(f"✅ Saved SHAP values to {out_path}")

if __name__ == "__main__":
    main()
