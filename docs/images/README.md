# Midterm Progress Report Images

This directory contains screenshots for the IE4198 Midterm Progress Report.

## Required Screenshots

### Figure 1: Main Dashboard Overview
- **File**: `dashboard_main.png`
- **Content**: Screenshot of the main dashboard page showing:
  - Forecasting chart with LightGBM vs LSTM predictions
  - SKU selector dropdown
  - Inventory simulator controls (service level slider)
  - Key metrics (cost savings, champion model, etc.)

### Figure 2: Multi-Page Navigation
- **File**: `dashboard_sidebar.png`
- **Content**: Screenshot showing the sidebar navigation with:
  - Project status section at bottom
  - Navigation menu (Dashboard, Sensitivity, Benchmark, Comparison)
  - Currently active page highlighted

### Figure 3: Sensitivity Analysis
- **File**: `sensitivity_analysis.png`
- **Content**: Screenshot of the sensitivity analysis page showing:
  - Interactive parameter sliders (holding cost, stockout cost, lead time, service level)
  - Cost impact analysis table
  - Sensitivity bar charts showing which parameters have highest impact

### Figure 4: Benchmark Results
- **File**: `benchmark_results.png`
- **Content**: Screenshot of the benchmark results page showing:
  - Model performance comparison table
  - Summary cards (champion, best RMSE, etc.)
  - Category breakdown (Classical, ML, DL)

### Figure 5: Model Comparison
- **File**: `model_comparison.png`
- **Content**: Screenshot of the model comparison page showing:
  - Detailed model cards with strengths/weaknesses
  - Decision framework section
  - Implementation recommendations

### Figure 6: Multi-Store Validation Results
- **File**: `multi_store_validation.png`
- **Content**: Screenshot or table showing:
  - Cross-store performance comparison (CA_1, CA_2, CA_3)
  - RMSE and cost metrics for each store
  - Performance variation analysis

### Figure 7: LSTM Architecture with Log Transform
- **File**: `lstm_log_transform.png`
- **Content**: Diagram or code snippet showing:
  - Log transformation approach: log(sales + 1)
  - How it addresses sparsity penalty
  - Before/after comparison or architecture diagram

## Image Specifications

- **Format**: PNG or JPG
- **Resolution**: High quality, readable text (at least 1920x1080)
- **Naming**: Use exact filenames as specified above
- **Size**: Keep under 2MB per image for report inclusion

## How to Capture Screenshots

1. Run the frontend: `cd frontend && pnpm dev`
2. Navigate to each page and capture full browser window
3. Use browser dev tools or screenshot tool to capture clean images
4. Crop unnecessary browser chrome if needed
5. Save with exact filenames in this directory