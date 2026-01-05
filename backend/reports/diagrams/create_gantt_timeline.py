from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for professional look
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def create_gantt():
    # 1. Define the Data (Project Timeline)
    data = [
        # --- PHASE I: IE 4197 (Fall 2025) ---
        {
            "Task": "WP1: Problem Definition & Scope",
            "Start": "2025-09-29",
            "End": "2025-10-17",
            "Phase": "Phase I: Definition",
            "Completion": 100,
        },
        {
            "Task": "WP2: Literature Review & EDA",
            "Start": "2025-10-20",
            "End": "2025-11-07",
            "Phase": "Phase I: Research",
            "Completion": 100,
        },
        {
            "Task": "WP3: Pipeline & Methodology Design",
            "Start": "2025-11-10",
            "End": "2025-12-12",
            "Phase": "Phase I: Design",
            "Completion": 100,
        },
        {
            "Task": "WP4: Pilot Benchmarking (10 Models)",
            "Start": "2025-12-15",
            "End": "2026-01-02",
            "Phase": "Phase I: Validation",
            "Completion": 100,
        },
        {
            "Task": "Final Report Submission (IE 4197)",
            "Start": "2026-01-05",
            "End": "2026-01-06",
            "Phase": "Milestone",
            "Completion": 100,
        },
        
        # --- PHASE II: IE 4198 (Spring 2026) ---
        {
            "Task": "WP5: Full-Scale Model Training",
            "Start": "2026-02-15",
            "End": "2026-03-15",
            "Phase": "Phase II: Execution",
            "Completion": 0,
        },
        {
            "Task": "WP6: Inventory Simulation & Optimization",
            "Start": "2026-03-16",
            "End": "2026-04-10",
            "Phase": "Phase II: Execution",
            "Completion": 0,
        },
        {
            "Task": "WP7: Dashboard Development (DSS)",
            "Start": "2026-04-01",
            "End": "2026-05-01",
            "Phase": "Phase II: Deployment",
            "Completion": 0,
        },
        {
            "Task": "WP8: Final Thesis & Defense",
            "Start": "2026-05-01",
            "End": "2026-05-20",
            "Phase": "Phase II: Closing",
            "Completion": 0,
        },
    ]

    # 2. Process Data
    df = pd.DataFrame(data)
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    df["Duration"] = (df["End"] - df["Start"]).dt.days

    # Color Palette
    phase_colors = {
        "Phase I: Definition": "#34495e",
        "Phase I: Research": "#3498db",
        "Phase I: Design": "#2980b9",
        "Phase I: Validation": "#1abc9c",
        "Milestone": "#e74c3c",
        "Phase II: Execution": "#f39c12",
        "Phase II: Deployment": "#d35400",
        "Phase II: Closing": "#8e44ad",
    }
    colors = [phase_colors[p] for p in df["Phase"]]

    # 3. Create Figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw Bars
    bars = ax.barh(
        y=df.index, 
        width=df["Duration"], 
        left=df["Start"], 
        height=0.5, 
        color=colors,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.5
    )

    # 4. Formatting
    ax.invert_yaxis()
    ax.set_yticks(df.index)
    ax.set_yticklabels(df["Task"], fontsize=11, fontweight='bold', color='#2c3e50')
    
    # X-Axis Dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=0, fontsize=10)

    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Title & Legend
    plt.title("Project Timeline: IE 4197 / IE 4198", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Timeline (2025-2026)", fontsize=12)

    # Add "Today" line
    today = pd.Timestamp("2026-01-05")
    plt.axvline(today, color='red', linestyle='--', alpha=0.8, label="Current Status")
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p) for p, c in phase_colors.items()]
    # Split legend into Phase I and II if needed, or keep simple
    # Let's just show Phase I vs II colors generally
    
    # 5. Save
    plt.tight_layout()
    output_dir = Path("backend/reports/diagrams/img")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "local_gantt_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Gantt Chart saved to: {output_file}")

if __name__ == "__main__":
    create_gantt()