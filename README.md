# **IE Capstone Project: Deep Learning for Supply Chain Forecasting**

## **📌 Project Overview**

This repository contains the code and data for the Industrial Engineering Capstone Project: **"A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization."**  
The project aims to demonstrate the financial and operational impact of using advanced AI (Deep Learning) forecasting methods compared to traditional statistical methods (ARIMA) on a large-scale retail dataset.

### **Key Objectives**

1. **Forecasting:** Benchmark **10+ competing models** across three tiers of complexity:
    -   **Classical:** ARIMA, Holt-Winters, ETS, Smoothing methods.
    -   **Machine Learning:** LightGBM, XGBoost, Prophet, Random Forest.
    -   **Deep Learning:** LSTM (Seq2Seq).
2.  **Optimization:** Drive a theoretical inventory policy (Reorder Point, Safety Stock) using these forecasts.
3.  **Comparison:** Quantify the financial impact ($) of model accuracy. **Goal achieved: +19.5% cost savings ($125k).**

## **📂 Repository Structure**

This is a **hybrid repository** containing both the Python Data Science environment and the Next.js Frontend application.  
ie-capstone-m5/  
├── backend/ \# 🐍 Python Backend (Data Science & AI)  
│ ├── notebooks/ \# Jupyter Notebooks for EDA and Prototyping  
│ ├── src/ \# Production Python scripts (Pipeline, Training)  
│ ├── data/ \# Data storage (Not committed to Git)  
│ └── models/ \# Saved model artifacts (Not committed to Git)  
│  
└── frontend/ \# ⚛️ Frontend Application (Next.js)  
 ├── src/ \# React components and pages  
 └── public/ \# Static assets

## **🚀 Getting Started**

### **Prerequisites**

-   **OS:** macOS (Apple Silicon M1/M2/M3 recommended) or Linux/Windows.
-   **Package Managers:** mamba (or conda) for Python, pnpm for Node.js.
-   **Git:** Version control.

### **1\. Python Environment Setup (Backend)**

We use a dedicated **Mamba/Conda** environment for reproducibility.  
\# 1\. Move to the backend directory  
cd backend

\# 2\. Create the environment (if you haven't already)  
mamba create \-n capstone python=3.10  
mamba activate capstone

\# 3\. Install Dependencies (MPS/Mac Optimized)  
mamba install pytorch torchvision torchaudio \-c pytorch  
mamba install \-c conda-forge jupyterlab numpy scikit-learn polars matplotlib seaborn lightgbm fastapi uvicorn statsmodels prophet xgboost

**Running Jupyter Notebooks:**  
\# Start Jupyter Lab  
jupyter lab

Downloading Data:  
We use the Kaggle M5 Forecasting \- Accuracy dataset.  
Ensure your Kaggle API key is in \~/.kaggle/kaggle.json.  
\# Run the download script (if available) or use the kaggle CLI  
cd data/raw  
kaggle competitions download \-c m5-forecasting-accuracy  
unzip m5-forecasting-accuracy.zip

### **2\. Frontend Setup (Dashboard)**

The frontend is a Next.js application used to visualize the forecast comparisons.  
\# 1\. Move to the frontend directory  
cd frontend

\# 2\. Install dependencies  
pnpm install

\# 3\. Run the development server  
pnpm dev

Open [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000) to view the dashboard.

## **🧠 Methodology & Stack**

### **Data Processing (Python)**

-   **Polars:** Used for high-performance data manipulation (the dataset has \~59M rows).
-   **Feature Engineering:** Lags, rolling windows, and categorical encoding of calendar events.

### **Modeling (The "Drag Race")**

We implemented and benchmarked **10 models** to find the champion:

1.  **Tier 1 (Classical):** Naive, SMA, WMA, SES, Holt Linear, Holt-Winters (Winner 🥈), ETS.
2.  **Tier 2 (ML Baselines):** XGBoost, Random Forest, Prophet, ARIMA / AutoARIMA.
3.  **Tier 3 (Deep Learning):**
    -   **LightGBM:** The overall **Accuracy & Financial Champion 🥇**.
    -   **LSTM:** A deep learning baseline (struggled with sparse data).

### **Optimization Results**

We simulated inventory for 3,049 items over 28 days.
-   **Baseline Cost (Naive):** $640,703
-   **LightGBM Cost:** $515,513
-   **Savings:** **$125,190 (19.5%)**

### **Visualization**

-   **Next.js 14 (App Router):** For the interactive dashboard.
-   **Recharts / D3:** For plotting forecast curves and confidence intervals.
-   **Tailwind CSS:** For styling.

## **👥 Team Members**

-   **AI & Software Lead:** \[Your Name\] \- (Architecture, PyTorch, Pipeline, Frontend)
-   **Optimization & Reporting Lead:** \[Name\] \- (Inventory Policy, Cost Analysis)
-   **IE Analyst:** \[Name\] \- (Classical Forecasting, EDA)
-   **Project Manager:** \[Name\] \- (Documentation, Presentations)

## **📄 License**

The source code is licensed under the MIT License.  
The final report and written content are licensed under CC BY-NC-SA 4.0.

## **📚 Documentation & Reports**

-   **[Final Report Template](./docs/Final_Report_Template.md):** The comprehensive technical report for this project.
-   **[Dataset Overview](./docs/dataset-overview.md):** Detailed breakdown of the M5 data structure.
-   **[Benchmark Report](./backend/reports/summaries/benchmark_report.md):** The results of the 10-model "drag race".
