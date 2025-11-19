# Backend ML & Data Pipeline

This directory contains all the Python code for the data processing, model training, forecasting, and optimization parts of the project.

## 📂 Directory Structure

The backend is organized into the following key directories:

```
backend/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   ├── diagrams/
│   │   └── img/
│   └── summaries/
├── results/
│   ├── forecasts/
│   └── optimization/
├── scripts/
│   ├── data_preparation/
│   ├── optimization/
│   ├── prediction/
│   └── training/
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
└── testing/
```

-   **`src`**: Contains the core, reusable Python modules for the project. This includes data loading, feature engineering, model definitions (PyTorch), and configuration.
-   **`scripts`**: Holds the high-level executable scripts that run the end-to-end pipeline. These scripts import and use the code from `src`.
-   **`data`**: The storage location for all data.
    -   `raw`: The original, immutable M5 competition data.
    -   `processed`: Cleaned, transformed, and feature-engineered data, often in a more efficient format like Parquet.
-   **`models`**: Stores saved model artifacts after training (e.g., `lstm_best.pt`, `baseline_lgbm.pkl`).
-   **`results`**: The destination for final pipeline outputs.
    -   `forecasts`: Contains the generated forecast CSV files (e.g., `forecast_lstm.csv`).
    -   `optimization`: Contains the final cost and accuracy comparisons (e.g., `optimization_summary.csv`).
-   **`reports`**: Contains human-readable reports, diagrams, and summaries about the experiments.
-   **`notebooks`**: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
-   **`testing`**: Holds scripts for validating the outputs of the pipeline, such as checking the format and integrity of generated forecast files.

## 🚀 Getting Started

### Prerequisites

-   **mamba** (or **conda**) for Python environment management.
-   **Python 3.10**

### Environment Setup

1.  **Navigate to the python directory:**
    _(This is a bit of a misnomer in the original root README, as the main directory is `backend`, not `python-ml`)_

    ```bash
    # From the project root
    cd backend/
    ```

2.  **Create and activate the Mamba environment:**

    ```bash
    mamba create -n capstone python=3.10
    mamba activate capstone
    ```

3.  **Install dependencies:**

    ```bash
    mamba install pytorch torchvision torchaudio -c pytorch
    mamba install -c conda-forge numpy scikit-learn polars matplotlib seaborn lightgbm fastapi uvicorn
    ```

## ⚙️ Running the Pipeline

The pipeline should be run in the following sequence. All commands are run from the project's root directory (`ie-capstone-m5/`).

### 1. Data Preparation

First, preprocess the raw data and generate features.

```bash
python backend/scripts/data_preparation/preprocess.py
python backend/scripts/data_preparation/make_features.py
```

### 2. Model Training

Train the machine learning and deep learning models.

```bash
# Train the LightGBM baseline
python backend/scripts/training/train_lgbm.py

# Train the LSTM model
python backend/scripts/training/train_lstm.py
```

### 3. Generate Forecasts

Use the trained models to generate the 28-day forecasts.

```bash
python backend/scripts/prediction/predict_lgbm.py
python backend/scripts/prediction/predict_lstm.py
```

### 4. Run Optimization Analysis

With the forecasts generated, run the optimization script to calculate inventory costs and compare the models' financial impact.

```bash
python backend/scripts/optimization/optimize.py
```

### 5. Validate Results

After running the pipeline, you can validate the outputs to ensure they meet the expected format and quality standards.

```bash
python backend/testing/validate_experiment_outputs.py
```
