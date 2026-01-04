# **Project Blueprint: Deep Learning for Supply Chain Forecasting**

**Project:** A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization  
**Your Role:** AI & Software Lead  
**Stack:** Python 3.10+, PyTorch 2.0+, Polars, LightGBM, FastAPI, Next.js 16, Docker (Optional)

---

## **1. Executive Summary & Key Findings**

This study compared three forecasting methodologies—Naive Baseline, Gradient Boosting (LightGBM), and Deep Learning (LSTM)—to determine their impact on Supply Chain Inventory Optimization using the M5 dataset.

**Key Finding:** While the Deep Learning (LSTM) model showed promising initial metrics during training, the **LightGBM (Gradient Boosting)** model provided the superior balance of accuracy and real-world cost-efficiency for this specific sparse retail dataset. LightGBM reduced total simulated logistics costs by **19.5%** compared to a naive baseline, representing a potential savings of over $125,000 in the experiment.

| Model        | RMSE (Error) | Total Logistics Cost | Performance vs. Naive      |
| :----------- | :----------- | :------------------- | :------------------------- |
| **Naive**    | 2.86         | $640,703             | Baseline                   |
| **LightGBM** | **2.10**     | **$515,513**         | **+19.5% Savings ($125k)** |
| **LSTM**     | 3.58         | $1,229,646           | -91% (Loss)                |

The LSTM's underperformance in the cost simulation, despite a strong validation Log-RMSE (0.53), is attributed to the **Sparsity Penalty**, where the model's predictions of small, non-zero values for items with many zero-sale days led to accumulating holding costs or significant stockout costs.

**Recommendation:** For this dataset, **LightGBM is the recommended production model** due to its robustness, efficiency, and proven cost savings. Future deep learning work should investigate probabilistic architectures (e.g., DeepAR) designed to handle highly intermittent demand.

---

## **2. Project Architecture & Structure**

**Objective:** Maintain a "Cookiecutter Data Science" structure that supports reproducibility, logging, and configuration management from Day 1.

### **2.1. Repository Structure (Hybrid Monorepo)**

-   **Root:** `ie-capstone-m5/`
-   **Backend:** `backend/` (Python/ML workspace)
-   **Frontend:** `frontend/` (Next.js Dashboard)

### **2.2. Detailed Backend Component Breakdown (`backend/`)**

-   **`src`**: Contains the core, reusable Python modules for the project. This includes data loading (`data/`), feature engineering (`features/`), model definitions (`models/`), configuration (`config/`), and utilities (`utils/`).
-   **`scripts`**: Holds the high-level executable scripts that run the end-to-end pipeline (e.g., `training/train_lstm.py`, `prediction/predict_lgbm.py`). These scripts import and use the code from `src`.
-   **`data`**: The storage location for all data.
    -   `raw/`: The original, immutable M5 competition data.
    -   `processed/`: Cleaned, transformed, and feature-engineered data, often in a more efficient format like Parquet.
-   **`models/`**: Stores saved model artifacts after training (e.g., `lstm_best.pt`, `baseline_lgbm.pkl`).
-   **`results/`**: The destination for final pipeline outputs.
    -   `forecasts/`: Generated forecast CSV files.
    -   `optimization/`: Final cost and accuracy comparisons.
-   **`reports/`**: Human-readable reports, diagrams, and experiment summaries.
-   **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
-   **`testing/`**: Scripts for validating the format and integrity of pipeline outputs.

---

## **3. End-to-End MLOps Pipeline**

### **3.1. Environment Setup**

The project uses a dedicated Mamba/Conda environment for reproducibility.

1.  **Navigate to the backend directory:**
    ```bash
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

### **3.2. Running the Pipeline**

The pipeline is executed via a series of scripts from the project root directory.

1.  **Data Preparation:** Preprocess raw data and generate features.
    ```bash
    python backend/scripts/data_preparation/preprocess.py
    python backend/scripts/data_preparation/make_features.py
    ```
2.  **Model Training:** Train the machine learning and deep learning models.
    ```bash
    python backend/scripts/training/train_lgbm.py
    python backend/scripts/training/train_lstm.py
    ```
3.  **Generate Forecasts:** Use trained models to generate 28-day forecasts.
    ```bash
    python backend/scripts/prediction/predict_lgbm.py
    python backend/scripts/prediction/predict_lstm.py
    ```
4.  **Run Optimization Analysis:** Calculate inventory costs to compare model impact.
    ```bash
    python backend/scripts/optimization/optimize.py
    ```
5.  **Validate Results:** Check outputs for format and quality standards.
    ```bash
    python backend/testing/validate_experiment_outputs.py
    ```

---

## **4. Modeling Strategy & Algorithm Comparison**

To provide a multi-faceted comparative analysis, we will conduct experiments along two parallel tracks, evaluating both algorithmic performance and the impact of data scale and tooling.

### **Track 1: IE Team Benchmark (Small-Scale, Traditional Tools)**

-   **Scope:** A random sample of 50-100 items from the dataset.
-   **Tools:** Excel, Minitab, or similar statistical software.
-   **Algorithms (Tier 1):**
    -   Naive Method
    -   Simple & Weighted Moving Average
    -   Single Exponential Smoothing
    -   Holt’s Linear Trend
    -   Holt-Winters Additive Seasonality
-   **Objective:** Establish a performance baseline that reflects traditional, small-scale analysis methods.

### **Track 2: AI/Dev Team Benchmark (Full-Scale, Python Pipeline)**

-   **Scope:** The complete dataset (~59 million rows).
-   **Tools:** Python (Statsmodels, Scikit-learn, PyTorch).
-   **Algorithms:** This track covers all three tiers of complexity to compare against the IE Team's baseline and determine the state-of-the-art.
    -   **Tier 1 (Classical Replication):** The same methods as the IE team, but applied at scale.
    -   **Tier 2 (Scalable Machine Learning):** ARIMA/SARIMA, Facebook Prophet, XGBoost.
    -   **Tier 3 (Deep Learning):** LightGBM, LSTM, and stretch goals like N-BEATS or a Transformer.
-   **Objective:** Quantify the performance gains from both superior algorithms and larger data.

### **Primary Analytical Goals**

This dual-track approach allows us to answer two key questions:

1.  **Which algorithm provides the lowest inventory cost?** (Absolute Performance)
2.  **What is the quantifiable value of scaling up?** We can directly compare the results of Tier 1 algorithms on small data (Track 1) vs. large data (Track 2) to measure the impact of data scale alone.

---

## **5. Phased Development Plan**

### **Phase 1: Data Engineering & Baseline**

-   **Goal:** Establish a high-performance data pipeline and a strong Machine Learning baseline.
-   **Tasks:**
    1.  **Data Ingestion (`preprocess.py`):** Ingest and optimize 59M rows using **Polars** for high-speed processing.
    2.  **Feature Engineering (`engineer.py`):** Create lag, rolling statistic, calendar, and price features.
    3.  **Baseline Model (`baseline.py`):** Train and validate the LightGBM model.
    4.  **Validation (`validate_experiment_outputs.py`):** Create scripts to assert the integrity of all pipeline outputs.

### **Phase 2: Deep Learning & MLOps**

-   **Goal:** Surpass the baseline using a custom PyTorch architecture.
-   **Tasks:**
    1.  **Custom Dataset (`dataset.py`):** Implement a `torch.utils.data.Dataset` class for efficiently feeding sequence data to the model.
    2.  **Model Architecture (`lstm.py`):** Design a custom LSTM/GRU model with embeddings for categorical features.
    3.  **Training Scripts (`train_lstm.py`):** Develop a robust training loop with logging, checkpointing, and mixed-precision support.
    4.  **Future Work:** Based on results, pivot to **Probabilistic Architectures** (e.g., DeepAR, N-BEATS) that can better model the zero-inflated nature of the data.

### **Phase 3: Inference & Optimization Analysis**

-   **Goal:** Generate forecasts and translate them into actionable business metrics.
-   **Tasks:**
    1.  **Recursive Inference (`prediction/*.py`):** Implement a loop to predict 28 days sequentially, using each prediction to update features for the next step.
    2.  **Optimization (`optimization/optimize.py`):** Simulate inventory policies (Reorder Point, Safety Stock) based on the generated forecasts to calculate and compare total logistics costs (holding + stockout).

### **Phase 4: Frontend Visualization**

-   **Goal:** Create an executive dashboard to communicate results effectively.
-   **Tech:** Next.js 16 (App Router), Recharts, Tailwind CSS.
-   **Components:**
    -   `ForecastChart.tsx`: Line chart comparing "Actuals," "LightGBM Forecast," and "LSTM Forecast."
    -   `InventorySimulator.tsx`: Sliders to adjust service level and see the impact on safety stock and total cost.

### **Phase 5: CI/CD & Deployment**

-   **Goal:** Automate testing and deployment.
-   **Tasks:**
    1.  **Continuous Integration (GitHub Actions):** On every push, automatically run linters (`ruff`) and validation scripts (`testing/*`) for the backend, and build/lint for the frontend.
    2.  **Deployment (Vercel):** Connect the GitHub repo to Vercel, with the root directory set to `frontend`. Forecast data will be pre-generated and committed to the repo for the frontend to consume statically.

---

## **6. Validation and Robustness Strategy**

To address the limitation of not having a formal hold-out test set, the following stress tests will be performed on the validation set results to ensure conclusions are robust.

1.  **Simulated Cross-Validation:** The error and cost will be calculated for each of the four weeks in the 28-day validation period. Consistent performance across all four weeks indicates a more robust model.
2.  **Naive Baseline Comparison:** The primary defense of model value. The financial savings of any proposed model will be benchmarked against a simple "Naive Forecast" (e.g., prediction = sales from 28 days ago). A significant margin of victory demonstrates true learning.
3.  **Sensitivity Analysis:** The optimization simulation will be re-run with artificially worsened forecasts (e.g., adding 10-20% random noise). If the recommended model still provides significant savings over the baseline, the conclusion is robust against potential overfitting.
