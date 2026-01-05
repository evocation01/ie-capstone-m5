import datetime
import os

import numpy as np
import pandas as pd

# --- PATH CONFIGURATION ---
# Based on project structure: backend/results/forecasts/ and backend/results/optimization/
BASE_DIR = os.path.join("backend", "results")
LOG_DIR = os.path.join("backend", "logs")
INPUT_FILE = os.path.join(BASE_DIR, "forecasts", "forecast_lgbm.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "optimization", "scaled_inventory_policy.csv")


def calculate_inventory_policy():
    """
    Calculates Safety Stock, Reorder Point, and Total Logistics Cost
    for all items based on LightGBM forecast error.
    """
    # 1. Load your forecast results
    if not os.path.exists(INPUT_FILE):
        print(
            f"Error: {INPUT_FILE} not found. Ensure the LightGBM forecast script has been run."
        )
        return

    df = pd.read_csv(INPUT_FILE)

    # 2. IE Parameters (Standard for Retail Capstones)
    service_level = 0.95  # 95% cycle service level
    z_score = 1.645  # z-value for 95%
    lead_time = 3  # 3-day lead time from distribution center
    holding_cost_rate = 0.01  # $0.01 per unit per day

    # 3. Calculate Error Scale (Using RMSE from optimization_summary.csv)
    # We use the global RMSE 2.10 as the standard deviation of forecast error (sigma_e).
    global_rmse = 2.1018

    # 4. IE Calculations
    # Demand during Lead Time (D_L): Sum of the first 'lead_time' days of forecast
    forecast_cols = [f"F{i}" for i in range(1, 29)]
    df["D_L"] = df[forecast_cols[:lead_time]].sum(axis=1)

    # Safety Stock (SS) = z * RMSE * sqrt(LeadTime)
    # Formula: SS = z * sigma_e * sqrt(L)
    df["safety_stock"] = z_score * global_rmse * np.sqrt(lead_time)

    # Reorder Point (r) = D_L + SS
    df["reorder_point"] = df["D_L"] + df["safety_stock"]

    # 5. Financial Impact Estimation
    # Expected Daily Holding Cost = Safety Stock * holding_rate
    df["daily_holding_cost"] = df["safety_stock"] * holding_cost_rate

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Export the full policy table
    output_cols = ["id", "D_L", "safety_stock", "reorder_point", "daily_holding_cost"]
    df[output_cols].to_csv(OUTPUT_FILE, index=False)

    # 6. Summary Stats & Logging
    total_ss = df["safety_stock"].sum()
    total_holding_annual = df["daily_holding_cost"].sum() * 365

    summary_text = (
        f"--- Scaled Inventory Optimization Summary ---\n"
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Model: LightGBM (RMSE={global_rmse})\n"
        f"Items Processed: {len(df)}\n"
        f"Total Safety Stock Units: {total_ss:,.0f}\n"
        f"Estimated Annual Holding Cost (LGBM): ${total_holding_annual:,.2f}\n"
        f"Output File: {OUTPUT_FILE}\n"
        f"---------------------------------------------\n"
    )

    print(summary_text)

    # Save to Log File
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = (
        f"optimization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    log_path = os.path.join(LOG_DIR, log_filename)

    with open(log_path, "w") as f:
        f.write(summary_text)

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    calculate_inventory_policy()
