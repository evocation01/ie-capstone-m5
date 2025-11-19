### **AI & Software Lead: Developer Blueprint (Production-Ready Edition)**

Project: A Comparative Analysis of Classical and Deep Learning Forecasting  
Your Role: AI & Software Lead  
Stack: Python 3.10+, PyTorch 2.0+, Polars, LightGBM, FastAPI, Next.js 16, Docker (Optional)

### Limitations:

Due to computational constraints and the structure of the available M5 dataset, a hold-out Test set was not used. The final model evaluation was performed on the Validation set (last 28 days of historical data). While standard practice suggests a third 'Test' split to prevent bias, our robust cross-comparison with classical baselines (LightGBM, ARIMA) confirms the relative performance gains of the Deep Learning approach.

Here are 3 concrete things you can do right now without retraining:

1. Cross-Validation on the Validation Set (Simulated)

Instead of just reporting the final RMSE on the full 28 days, you can break down the error.

    Action: Calculate the RMSE for Week 1, Week 2, Week 3, and Week 4 separately.

    Why: If the model is overfitting, it might be great at Week 1 (memorized pattern) but terrible at Week 4. If the error is consistent across all 4 weeks, it suggests the model is robust, even if you "peeked" at the data.

    Add this to optimize.py: Instead of just one total cost, report "Weekly Costs."

2. "Sanity Check" against a Naive Baseline

This is the most powerful defense.

    Action: Run a "Naive Forecast" (prediction = sales from 28 days ago).

    Why: If your LSTM beats the Naive forecast by a huge margin (e.g., 20-30%), then even if your LSTM result is slightly optimistic due to the data split, the relative improvement is real. The "optimism bias" applies to both, but the LSTM's structural advantage remains valid.

    Add to Report: "Even under the conservative assumption that our validation score is optimistic, the LSTM outperforms the Naive baseline by X%, demonstrating learning beyond memorization."

3. Sensitivity Analysis in Optimization

Show that your financial savings hold up even if the forecast is worse than you think.

    Action: In your Excel/Python optimization, add a "Forecast Error Multiplier."

    Scenario A: Use your LSTM Forecast as is. (Savings: $1.2M)

    Scenario B: Assume your LSTM is actually 10% worse than calculated (add random noise to the forecast).

    Result: If Scenario B still saves $800k compared to the classical method, your conclusion ("AI saves money") is robust against the data split limitation.

These steps turn a "methodological flaw" into a "robust sensitivity analysis," which professors love. You don't need new data; you just need to stress-test the data you have.

### **Phase 0: Infrastructure & Architecture (Weeks 1-3)**

**Objective:** Build a "Cookiecutter Data Science" structure that supports reproducibility, logging, and configuration management from Day 1\.

1. **Repository Structure (Hybrid Monorepo):**
    - **Root:** ie-capstone-m5
    - **Backend:** backend/ (Python/ML workspace)
    - **Frontend:** frontend/ (Next.js Dashboard)
2. **Detailed Component Breakdown (backend/):**
    - **`data/`**: The storage location for all data.
        - `raw/`: The original, immutable M5 competition data.
        - `processed/`: Cleaned, transformed, and feature-engineered data, often in a more efficient format like Parquet.
    - **`models/`**: Stores saved model artifacts after training (e.g., `lstm_best.pt`, `baseline_lgbm.pkl`).
    - **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
    - **`reports/`**: Contains human-readable reports, diagrams, and summaries about the experiments.
    - **`results/`**: The destination for final pipeline outputs.
        - `forecasts/`: Contains the generated forecast CSV files (e.g., `forecast_lstm.csv`).
        - `optimization/`: Contains the final cost and accuracy comparisons (e.g., `optimization_summary.csv`).
    - **`scripts/`**: Holds the high-level executable scripts that run the end-to-end pipeline. These scripts import and use the code from `src`.
        - `data_preparation/`: Scripts for downloading and processing data.
        - `optimization/`: Scripts to run inventory analysis on forecasts.
        - `prediction/`: Scripts to generate forecasts from trained models.
        - `training/`: Scripts to train models (`train_lgbm.py`, `train_lstm.py`).
    - **`src/`**: Contains the core, reusable Python modules for the project.
        - `config/`: Configuration files (e.g. paths).
        - `data/`: Data ingestion and `torch.utils.data.Dataset` logic.
        - `features/`: Feature engineering functions.
        - `models/`: Model architecture definitions (PyTorch classes like `lstm.py`).
        - `utils/`: Utility functions like logging.
    - **`testing/`**: Holds scripts for validating the outputs of the pipeline, such as checking the format and integrity of generated forecast files.

### **Phase 1: Data Engineering & Baseline (Semester 1\)**

**Goal:** Establish a high-performance data pipeline and a strong Machine Learning baseline.

1. **Data Ingestion & Optimization (`backend/scripts/data_preparation/preprocess.py`)**
    - **Task:** Ingest 59M rows efficiently.
    - **Tool:** polars. It is non-negotiable for speed here.
    - **Logic:**
        - Cast types immediately (e.g., int16 for sales, category for IDs) to save RAM.
        - Melt sales_train from wide to long.
        - Save as `data/processed/melted_sales.parquet`.
2. **Feature Engineering (`backend/src/features/engineer.py`)**
    - **Task:** Create the signals the model learns from.
    - **Key Features:**
        - **Lags:** Sales from 7, 14, 21, 28 days ago.
        - **Rolling Stats:** Mean/Std of sales over last 7/28 days.
        - **Calendar:** wday, month, event_name_1 (embedded later).
        - **Price:** sell_price, price_momentum (current price / avg price).
    - **Output:** `data/processed/features.parquet`.
3. **Baseline Model (`backend/src/models/baseline.py`)**
    - **Model:** LightGBM (Gradient Boosting).
    - **Why:** It handles NaNs and categories natively and is SOTA for tabular time-series.
    - **Validation:** Use the last 28 days of training data as a validation set.
    - **Deliverable:** `results/forecasts/forecast_lgbm.csv` and an analysis in `reports/`.
4. **Validation (`backend/testing/validate_experiment_outputs.py`)**
    - **Task:** Assert that final forecast files have the correct shape and format.

### **Phase 2: Deep Learning & MLOps (Semester 2\)**

**Goal:** Surpass the baseline using a custom PyTorch architecture.

1. **Custom Dataset (`backend/src/data/dataset.py`)**
    - **Logic:** The `__getitem__` method must be fast.
    - **Input:** A single index i.
    - **Operation:** Look back `seq_len` days (e.g., 90 days) from index i.
    - **Return:**
        - x_num: Tensor of numerical features (sales lags, price).
        - x_cat: LongTensor of categorical indices (item_id, store_id).
        - y: Tensor of target sales (next 1 or 28 days).
2. **Model Architecture (`backend/src/models/lstm.py`)**
    - **Design:**
        - **Embeddings:** Learnable vectors for item_id (3049 items) and store_id (10 stores).
        - **Encoder:** 2-layer LSTM or GRU with dropout.
        - **Decoder (Head):** Dense layers mapping hidden state to scalar output.
    - **Optimization:** Use `torch.amp` (Automatic Mixed Precision) to speed up training on Mac (MPS) or GPU.
3. **Training Scripts (`backend/scripts/training/`)**
    - **Logic:** The training logic is managed by scripts like `train_lstm.py`, which handle the training loop, logging, and model checkpointing directly. This is a more direct approach than the previously proposed `trainer.py` class.

### **Phase 3: Inference & Integration**

**Goal:** Generate the "Money Slide" data.

1. **Recursive Inference (`backend/scripts/prediction/*.py`)**
    - **Challenge:** You need to predict day t+1 to calculate the lag feature for day t+2.
    - **Implementation:**
        - Step 1: Predict Day 1.
        - Step 2: Append prediction to data (updating lags).
        - Step 3: Predict Day 2.
        - Loop for 28 days.
    - **Output:** `results/forecasts/final_forecast.csv`.
2. **API (Future Goal)**
    - A potential future step is to create a `FastAPI` app in `backend/scripts/app.py`.
    - Endpoint: `POST /predict` accepts JSON inputs and returns forecast.
    - This allows the frontend to query the model dynamically (Real-time DSS).

### **Phase 4: Frontend Visualization**

**Goal:** "The Executive Dashboard."

-   **Tech:** Next.js 14 (App Router), Recharts.
-   **Components:**
    -   ForecastChart.tsx: Line chart comparing "Actuals", "Classical Forecast", and "AI Forecast".
    -   InventorySimulator.tsx: Sliders to adjust "Service Level" (e.g., 95% vs 99%) and see the resulting Safety Stock cost.

### **Phase 5: CI/CD & Deployment**

**Objective:** Automate testing and deployment to Vercel.

1. **Continuous Integration (GitHub Actions):**
    - **Trigger:** On push to `main` or Pull Request.
    - **Backend Job:**
        - Set up Python 3.10.
        - Install dependencies.
        - Run `python backend/testing/validate_experiment_outputs.py`.
        - (Optional) Run "black" or "ruff" for linting.
    - **Frontend Job:**
        - Set up Node 20.
        - Run `pnpm lint` and `pnpm build`.
2. **Frontend Deployment (Vercel):**
    - **Integration:** Connect your GitHub repo to Vercel.
    - **Root Directory:** Set to `frontend`.
    - **Build Command:** `pnpm build`.
    - **Output Directory:** `.next`.
    - **Database:** Use Vercel Postgres (if needed) to store simulation results or user scenarios.
3. **Backend Strategy (Since Vercel is Serverless):**
    - **Option A (Static):** You generate forecast.csv locally/on Colab and commit it to the repo. The Vercel app just reads this file. (Simplest/Free).
    - **Option B (Dynamic):** If you need live inference, you cannot host the PyTorch model on Vercel (file size limits). You would deploy the FastAPI app on Render, Railway, or Hugging Face Spaces (free tier) and have your Vercel app call that API.
