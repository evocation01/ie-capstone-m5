import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import polars as pl

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "backend/results"
DATA_DIR = BASE_DIR / "backend/data/processed"
RAW_DATA_DIR = BASE_DIR / "backend/data/raw"
IMG_DIR = BASE_DIR / "backend/reports/diagrams/img/final_report"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def plot_zero_inflation():
    """Figure 4.1: Histogram of Sales (0 vs non-0)"""
    print("Generating Figure 4.1 (Zero Inflation)...")
    try:
        # Lazy scan and fetch random sample
        q = pl.scan_parquet(DATA_DIR / "melted_sales.parquet")
        # Fetch first 1M rows (random enough for this distribution)
        sample = q.collect().head(1000000).select("sales").to_pandas()
        
        plt.figure(figsize=(10, 5))
        # Count 0s vs others
        counts = sample['sales'].value_counts().sort_index().head(10) # 0 to 9
        
        sns.barplot(x=counts.index, y=counts.values, color="skyblue", edgecolor="black")
        plt.title("Figure: Zero-Inflation in M5 Dataset (Sample of 1M Rows)")
        plt.xlabel("Daily Unit Sales")
        plt.ylabel("Frequency")
        plt.yscale("log") # Log scale to show the drop
        plt.savefig(IMG_DIR / "fig_4_1_zero_inflation.png", bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 4.1: {e}")

def plot_benchmark_leaderboard():
    """Figure 4.2: RMSE Bar Chart"""
    print("Generating Figure 4.2 (Benchmark Leaderboard)...")
    try:
        df = pd.read_csv(RESULTS_DIR / "benchmark/benchmark_results_full.csv")
        
        # Aggregation
        leaderboard = df.groupby("model")["rmse"].mean().sort_values()
        
        plt.figure(figsize=(12, 6))
        colors = ['#2ecc71' if x == leaderboard.min() else '#e74c3c' if x == leaderboard.max() else '#3498db' for x in leaderboard.values]
        
        ax = sns.barplot(x=leaderboard.values, y=leaderboard.index, palette=colors)
        plt.title("Figure: Final Benchmark Leaderboard (Store CA_1)")
        plt.xlabel("Average RMSE (Lower is Better)")
        
        # Add labels
        for i, v in enumerate(leaderboard.values):
            ax.text(v + 0.02, i, f"{v:.4f}", va='center')
            
        plt.savefig(IMG_DIR / "fig_4_2_leaderboard.png", bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 4.2: {e}")

def plot_financial_impact():
    """Figure 5.3: Cost Comparison"""
    print("Generating Figure 5.3. (Financial Impact)...")
    try:
        df = pd.read_csv(RESULTS_DIR / "optimization/optimization_summary.csv")
        
        # Sort by Total Cost
        df = df.sort_values("Total_Cost")
        
        plt.figure(figsize=(10, 6))
        
        models = df['Model']
        holding = df['Holding_Cost']
        stockout = df['Stockout_Cost']
        
        # Stacked Bar
        p1 = plt.bar(models, stockout, color='#e74c3c', label='Stockout Cost')
        p2 = plt.bar(models, holding, bottom=stockout, color='#f1c40f', label='Holding Cost')
        
        plt.title("Figure: Financial Impact Analysis (Inventory Simulation)")
        plt.ylabel("Total Logistics Cost ($)")
        plt.legend()
        
        # Annotate bars
        for i, model in enumerate(models):
            total = holding.iloc[i] + stockout.iloc[i]
            plt.text(i, total + (total*0.05), f"${total:,.0f}", ha='center', weight='bold')

        plt.savefig(IMG_DIR / "fig_4_4_financial_impact.png", bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 4.4: {e}")

def plot_sparsity_penalty():
    """Figure 4.3: Line Chart (Actual vs Predicted)"""
    print("Generating Figure 4.3 (Sparsity Penalty)...")
    try:
        lgbm = pd.read_csv(RESULTS_DIR / "forecasts/forecast_lgbm.csv").set_index("id")
        lstm = pd.read_csv(RESULTS_DIR / "forecasts/forecast_lstm.csv").set_index("id")
        
        # We need actuals for the last 28 days.
        raw = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv")
        actual_cols = [c for c in raw.columns if c.startswith("d_")][-28:]
        
        # Ensure candidates are in the forecast index
        available_items = lgbm.index
        candidates = [i for i in available_items if "HOBBIES" in i]
        
        target_item = None
        for cand in candidates: # Iterate through all HOBBIES candidates
            actuals = raw[raw['id'] == cand][actual_cols].values.flatten()
            # Look for a mix of zeros and spikes (not just all zeros)
            zeros = (actuals == 0).sum()
            if zeros > 10 and zeros < 25: # Sparse but active
                target_item = cand
                break
                
        if not target_item:
            target_item = lgbm.index[0]            
            
        print(f"Selected item for sparsity plot: {target_item}")
        
        y_true = raw[raw['id'] == target_item][actual_cols].values.flatten()
        y_lgbm = lgbm.loc[target_item, [f"F{i}" for i in range(1,29)]].values.flatten()
        y_lstm = lstm.loc[target_item, [f"F{i}" for i in range(1,29)]].values.flatten()
        
        days = range(1, 29)
        
        plt.figure(figsize=(12, 5))
        plt.plot(days, y_true, 'k-o', label='Actual Sales', linewidth=2)
        plt.plot(days, y_lstm, 'r--', label='LSTM (Deep Learning)', alpha=0.8)
        plt.plot(days, y_lgbm, 'g-', label='LightGBM (ML)', alpha=0.8)
        
        plt.title(f"Figure: The Sparsity Penalty (Item: {target_item})")
        plt.xlabel("Day of Forecast")
        plt.ylabel("Unit Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(IMG_DIR / "fig_4_3_sparsity_penalty.png", bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 4.3: {e}")

if __name__ == "__main__":
    plot_zero_inflation()
    plot_benchmark_leaderboard()
    plot_financial_impact()
    plot_sparsity_penalty()
    print("✅ All plots generated in backend/reports/diagrams/img/final_report/")