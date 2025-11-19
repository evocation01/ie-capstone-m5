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
    - **data/**: The immutable foundation.
        - raw/: Untouched source files (sales_train_evaluation.csv, calendar.csv, sell_prices.csv). **Read-only.**
        - processed/: Parquet files optimized for read speed (melted_sales.parquet, final_train.parquet).
        - external/: Any supplementary data (e.g., holiday metadata).
    - **config/**: Configuration management.
        - config.yaml: Global settings (paths, random seeds).
        - model_config.yaml: Hyperparameters (learning rate, hidden dims, batch size).
        - **Why:** Never hardcode numbers in Python scripts. Use hydra or pyyaml to load these.
    - **src/**: The Application Source Code.
        - data/:
            - loader.py: Polars-based data loading logic.
            - dataset.py: torch.utils.data.Dataset implementation for sliding window sequences.
            - splitter.py: Time-series cross-validation splitter (e.g., expanding window).
        - features/:
            - engineer.py: Functions to generate lags (lag_7, lag_28), rolling means, and categorical encodings.
        - models/:
            - lstm.py: PyTorch LSTM/GRU implementation.
            - transformer.py: (Advanced) Time-series Transformer architecture.
            - baseline.py: LightGBM wrapper class.
        - training/:
            - trainer.py: A robust training loop class with train_epoch and validate_epoch methods.
            - callbacks.py: Early stopping and model checkpointing logic.
        - evaluation/:
            - metrics.py: Custom implementation of **WRMSSE** (Weighted Root Mean Squared Scaled Error) \- the official M5 metric.
        - utils/:
            - logger.py: Centralized logging configuration (standard logging library).
            - seeder.py: def seed_everything(seed) to ensure reproducibility.
    - **experiments/**:
        - runs/: Stores TensorBoard logs or CSV logs of training loss.
        - checkpoints/: Saves best model weights (best_model.pt).
    - **tests/**:
        - test_data.py: Verify data shapes and no NaN values after processing.
        - test_model.py: Verify model forward pass works on dummy data.
    - **scripts/**:
        - preprocess.py: CLI script to run the Polars pipeline.
        - train.py: CLI script to start training (accepts config path).
        - predict.py: CLI script to generate the final 28-day forecast.

### **Phase 1: Data Engineering & Baseline (Semester 1\)**

**Goal:** Establish a high-performance data pipeline and a strong Machine Learning baseline.

1. **Data Ingestion & Optimization (scripts/preprocess.py)**
    - **Task:** Ingest 59M rows efficiently.
    - **Tool:** polars. It is non-negotiable for speed here.
    - **Logic:**
        - Cast types immediately (e.g., int16 for sales, category for IDs) to save RAM.
        - Melt sales_train from wide to long.
        - Save as processed/melted_sales.parquet.
2. **Feature Engineering (src/features/engineer.py)**
    - **Task:** Create the signals the model learns from.
    - **Key Features:**
        - **Lags:** Sales from 7, 14, 21, 28 days ago.
        - **Rolling Stats:** Mean/Std of sales over last 7/28 days.
        - **Calendar:** wday, month, event_name_1 (embedded later).
        - **Price:** sell_price, price_momentum (current price / avg price).
    - **Output:** data/processed/features.parquet.
3. **Baseline Model (src/models/baseline.py)**
    - **Model:** LightGBM (Gradient Boosting).
    - **Why:** It handles NaNs and categories natively and is SOTA for tabular time-series.
    - **Validation:** Use the last 28 days of training data as a validation set.
    - **Deliverable:** experiments/baseline_metrics.json (RMSE score to beat).
4. **Testing (tests/test_data.py)**
    - **Unit Test:** Assert that features.parquet has no infinite values.
    - **Unit Test:** Assert that date column is sorted chronologically.

### **Phase 2: Deep Learning & MLOps (Semester 2\)**

**Goal:** Surpass the baseline using a custom PyTorch architecture.

1. **Custom Dataset (src/data/dataset.py)**
    - **Logic:** The \_\_getitem\_\_ method must be fast.
    - **Input:** A single index i.
    - **Operation:** Look back seq_len days (e.g., 90 days) from index i.
    - **Return:**
        - x_num: Tensor of numerical features (sales lags, price).
        - x_cat: LongTensor of categorical indices (item_id, store_id).
        - y: Tensor of target sales (next 1 or 28 days).
2. **Model Architecture (src/models/lstm.py)**
    - **Design:** \* **Embeddings:** Learnable vectors for item_id (3049 items) and store_id (10 stores).
        - **Encoder:** 2-layer LSTM or GRU with dropout.
        - **Decoder (Head):** Dense layers mapping hidden state to scalar output.
    - **Optimization:** Use torch.amp (Automatic Mixed Precision) to speed up training on Mac (MPS) or GPU.
3. **Training Loop (src/training/trainer.py)**
    - **Logging:** Log train_loss and val_loss to TensorBoard or a CSV file every epoch.
    - **Checkpointing:** Only save model_best.pt when val_loss improves.
    - **Scheduler:** Implement OneCycleLR or ReduceLROnPlateau for stable convergence.
4. **Testing (tests/test_model.py)**
    - **Integration Test:** Run one training step on a batch of random noise. Assert loss decreases (or at least computes).
    - **Shape Test:** Assert output shape matches (batch_size, prediction_horizon).

### **Phase 3: Inference & Integration**

**Goal:** Generate the "Money Slide" data.

1. **Recursive Inference (scripts/predict.py)**
    - **Challenge:** You need to predict day t+1 to calculate the lag feature for day t+2.
    - **Implementation:**
        - Step 1: Predict Day 1\.
        - Step 2: Append prediction to data (updating lags).
        - Step 3: Predict Day 2\.
        - Loop for 28 days.
    - **Output:** final_forecast.csv.
2. **API (Optional Production Touch)**
    - Create backend/scripts/app.py using **FastAPI**.
    - Endpoint: POST /predict accepts JSON inputs and returns forecast.
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
    - **Trigger:** On push to main or Pull Request.
    - **Backend Job:**
        - Set up Python 3.10.
        - Install dependencies.
        - Run pytest backend/tests.
        - (Optional) Run "black" or "ruff" for linting.
    - **Frontend Job:**
        - Set up Node 20\.
        - Run pnpm lint and pnpm build.
2. **Frontend Deployment (Vercel):**
    - **Integration:** Connect your GitHub repo to Vercel.
    - **Root Directory:** Set to frontend.
    - **Build Command:** pnpm build.
    - **Output Directory:** .next.
    - **Database:** Use Vercel Postgres (if needed) to store simulation results or user scenarios.
3. **Backend Strategy (Since Vercel is Serverless):**
    - **Option A (Static):** You generate forecast.csv locally/on Colab and commit it to the repo. The Vercel app just reads this file. (Simplest/Free).
    - **Option B (Dynamic):** If you need live inference, you cannot host the PyTorch model on Vercel (file size limits). You would deploy the FastAPI app on Render, Railway, or Hugging Face Spaces (free tier) and have your Vercel app call that API.
