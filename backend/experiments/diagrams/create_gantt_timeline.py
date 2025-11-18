from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def create_gantt():
    # 1. Define the Data (Your exact Timeline)
    data = [
        # SECTION: IE 4197 (Current)
        {
            "Task": "WP1: Problem Definition",
            "Start": "2025-09-29",
            "End": "2025-10-17",
            "Status": "Done",
        },
        {
            "Task": "WP2: Literature & Data EDA",
            "Start": "2025-10-20",
            "End": "2025-11-07",
            "Status": "Done",
        },
        {
            "Task": "WP3: Theoretical Model Design",
            "Start": "2025-11-24",
            "End": "2025-12-12",
            "Status": "Active",
        },
        {
            "Task": "WP4: Data Verification & Tools",
            "Start": "2025-12-15",
            "End": "2026-01-02",
            "Status": "Planned",
        },
        {
            "Task": "Final Report Submission",
            "Start": "2026-01-05",
            "End": "2026-01-06",
            "Status": "Critical",
        },  # 1 day duration
        # SECTION: IE 4198 (Planned) - Added a spacer for visual separation if needed, or just continuous
        {
            "Task": "Model Training (AI & Classic)",
            "Start": "2026-02-15",
            "End": "2026-03-30",
            "Status": "Planned",
        },
        {
            "Task": "Inventory Optimization Logic",
            "Start": "2026-03-01",
            "End": "2026-04-15",
            "Status": "Planned",
        },
        {
            "Task": "Dashboard Integration",
            "Start": "2026-04-01",
            "End": "2026-05-01",
            "Status": "Planned",
        },
        {
            "Task": "Final Thesis Writing",
            "Start": "2026-04-15",
            "End": "2026-05-20",
            "Status": "Planned",
        },
    ]

    # 2. Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    df["Duration"] = df["End"] - df["Start"]

    # Calculate duration in days for matplotlib
    df["Duration_Days"] = df["Duration"].dt.days

    # 3. Define Colors based on Status
    color_map = {
        "Done": "#bdc3c7",  # Grey
        "Active": "#3498db",  # Blue
        "Planned": "#2ecc71",  # Green
        "Critical": "#e74c3c",  # Red
    }
    colors = [color_map[status] for status in df["Status"]]

    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create horizontal bars
    # (y, width, left, height) -> (index, duration, start_date, thickness)
    bars = ax.barh(
        df.index, df["Duration_Days"], left=df["Start"], height=0.6, color=colors
    )

    # 5. Formatting

    # Y-Axis: Show Task Names
    ax.set_yticks(df.index)
    ax.set_yticklabels(df["Task"], fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Put the first task at the top

    # X-Axis: Date Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    # Grid and Labels
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_xlabel("Timeline")
    ax.set_title(
        "IE 4197/4198 Project Timeline", fontsize=14, fontweight="bold", pad=20
    )

    # Add a Legend manually
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map["Done"], label="Done"),
        Patch(facecolor=color_map["Active"], label="Active"),
        Patch(facecolor=color_map["Planned"], label="Planned"),
        Patch(facecolor=color_map["Critical"], label="Critical Deadline"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # 6. Save
    plt.tight_layout()
    output_file = "local_gantt_chart.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Success! Chart saved locally as: {output_file}")

    # Optional: Show plot window
    # plt.show()


if __name__ == "__main__":
    create_gantt()
