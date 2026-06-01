import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add backend root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths

def main():
    print("🚀 Exporting frontend data...")

    # Items to export
    top_ids = [
        'FOODS_3_090_CA_1_validation', 
        'FOODS_3_586_CA_1_validation', 
        'FOODS_3_120_CA_1_validation', 
        'FOODS_3_252_CA_1_validation', 
        'FOODS_3_501_CA_1_validation'
    ]

    # 1. Load Ground Truth
    df_actual = pd.read_csv(paths.RAW_DATA_DIR / "sales_train_validation.csv")
    
    # 2. Get history (last 90 days before validation)
    # Validation starts at d_1914, but our raw data ends at d_1913
    # The last 28 days (d_1886 to d_1913) are the validation actuals
    # History would be d_1796 to d_1885 (90 days)
    all_d_cols = [c for c in df_actual.columns if c.startswith('d_')]
    validation_cols = all_d_cols[-28:]
    history_cols = all_d_cols[-118:-28] # 90 days of history + 28 validation

    data = {}

    for id in top_ids:
        row = df_actual[df_actual['id'] == id]
        if row.empty: continue
        
        history = row[history_cols].values[0].tolist()
        actual = row[validation_cols].values[0].tolist()
        
        data[id] = {
            "history": history,
            "actual": actual,
            "forecasts": {}
        }

    # 3. Load Model Forecasts
    forecast_files = {
        "LightGBM": paths.FORECASTS_DIR / "forecast_lgbm.csv",
        "LSTM": paths.FORECASTS_DIR / "forecast_lstm.csv",
        "DeepAR": paths.FORECASTS_DIR / "forecast_deepar.csv",
        "Naive": None # We'll generate it from history
    }

    # Naive forecast
    for id in top_ids:
        if id in data:
            # Naive is just the last 28 days of history repeated
            data[id]["forecasts"]["Naive"] = data[id]["history"][-28:]

    for model_name, path in forecast_files.items():
        if path and path.exists():
            df_fc = pd.read_csv(path)
            f_cols = [c for c in df_fc.columns if c.startswith('F')]
            for id in top_ids:
                if id in data:
                    fc_row = df_fc[df_fc['id'] == id]
                    if not fc_row.empty:
                        data[id]["forecasts"][model_name] = fc_row[f_cols].values[0].tolist()
        elif path:
            print(f"Warning: {model_name} forecast not found at {path}")

    # 4. Export to JSON
    output_dir = project_root.parent / "frontend" / "public" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "sku_data.json", "w") as f:
        json.dump(data, f)
    
    # 5. Export Summary
    summary_path = paths.OPTIMIZATION_DIR / "optimization_summary.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        summary_list = df_summary.to_dict(orient="records")
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary_list, f)
            
    # 6. Export SHAP Explainability
    shap_path = paths.RESULTS_DIR / "explainability" / "shap_importance.json"
    if shap_path.exists():
        with open(shap_path, "r") as f:
            shap_data = json.load(f)
        with open(output_dir / "shap_importance.json", "w") as f:
            json.dump(shap_data, f)
            
    print(f"✅ Exported data to {output_dir}")

if __name__ == "__main__":
    main()
