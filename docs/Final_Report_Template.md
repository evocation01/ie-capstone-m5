# IE Capstone Project: Final Report Template

**Project Title:** Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization  
**Course:** IE 4197 - Engineering Project I/II  
**Date:** January 2026  
**Team Members:** [Your Names Here]

---

## 📄 Abstract
*(Approx. 250 words)*
Provide a high-level summary of the entire project.
-   **Problem:** Retail inventory optimization is challenging due to sparse demand and high seasonality.
-   **Methodology:** We compared **10 forecasting models** across three tiers (Classical, Machine Learning, Deep Learning) using the M5 Walmart dataset (30,490 time series).
-   **Key Results:**
    -   **Accuracy:** LightGBM (RMSE 1.41) and Holt-Winters (RMSE 1.45) outperformed advanced Deep Learning models (LSTM RMSE 2.02).
    -   **Financial Impact:** Implementing the LightGBM model reduced total logistics costs by **19.5% ($125,190)** compared to a naive baseline.
-   **Conclusion:** Simpler, interpretable models (Holt-Winters) or efficient ML (LightGBM) are often superior to complex Deep Learning for sparse retail data.

---

## 1. Introduction

### 1.1 Project Background
-   Supply chain efficiency relies on accurate demand forecasting.
-   Over-forecasting leads to high holding costs; under-forecasting leads to stockouts and lost revenue.
-   The retail industry faces "The Sparsity Problem" (many items have 0 sales on most days).

### 1.2 Objectives
1.  **Forecast:** Develop and benchmark models ranging from simple smoothing to Deep Learning.
2.  **Optimize:** Simulate an inventory policy (Reorder Point, Safety Stock) to translate accuracy into dollars.
3.  **Compare:** Quantify the trade-off between model complexity and business value.

### 1.3 Dataset Description (The M5 Data)
-   **Source:** Walmart (via Kaggle/Makridakis Open Forecasting Center).
-   **Scope:** 3,049 items × 10 stores × 1,913 days (~5 years).
-   **Features:** Hierarchical (State > Store > Dept), Calendar Events (Super Bowl), Prices.
-   *See `docs/dataset-overview.md` for full details.*

---

## 2. Methodology

### 2.1 Data Processing Pipeline
-   **Preprocessing:** `Polars` was used for high-performance manipulation of the 50M+ row dataset.
-   **Feature Engineering:**
    -   **Lags:** 7-day, 28-day lags.
    -   **Rolling Windows:** Moving averages to capture trends.
    -   **Calendar:** Encoding of "Special Events" and "SNAP" (food stamp) days.

### 2.2 Forecasting Models (The "Drag Race")
We implemented a multi-tier benchmarking strategy:

#### Tier 1: Classical Methods (Statistical)
-   **Naive:** Baseline (Last observed value).
-   **SMA / WMA:** Simple/Weighted Moving Averages.
-   **SES:** Single Exponential Smoothing (Level only).
-   **Holt-Winters (Winner 🥈):** Captures Level + Trend + Seasonality.
-   **ETS:** State-space implementation of exponential smoothing.

#### Tier 2: Machine Learning Baselines
-   **ARIMA:** Auto-Regressive Integrated Moving Average.
-   **Prophet:** Facebook's additive regression model (handles holidays well).
-   **XGBoost:** Gradient Boosting on lag features.

#### Tier 3: Advanced Deep Learning
-   **LightGBM (Winner 🥇):** Highly efficient gradient boosting decision tree.
-   **LSTM:** Long Short-Term Memory (Recurrent Neural Network) for sequence modeling.

### 2.3 Inventory Optimization Strategy
We utilized a standard `(Q, r)` inventory policy:
-   **Safety Stock (SS):** Buffer against forecast error (RMSE).
-   **Reorder Point (r):** Trigger replenishment when stock < Lead Time Demand + SS.
-   **Cost Function:** `Total Cost = Holding Cost + Stockout Cost`.

---

## 3. Results & Analysis

### 3.1 Forecast Accuracy Leaderboard (RMSE)
*Tested on 3,049 items over a 28-day horizon.*

| Rank | Model | RMSE | Performance |
| :--- | :--- | :--- | :--- |
| **1** | **LightGBM** | **1.4121** | **Best** |
| 2 | Holt-Winters | 1.4533 | Strong |
| 3 | SMA (28-day) | 1.4536 | Good |
| ... | ... | ... | ... |
| 8 | LSTM | 2.0234 | Poor |

*Analysis: LightGBM wins due to its ability to segment zero-days using tree splits. Holt-Winters remains extremely competitive due to strong weekly seasonality in retail.*

### 3.2 Financial Impact (The "Real" Result)
RMSE is abstract; Dollars are real.

| Model | Total Cost | Savings vs. Baseline |
| :--- | :--- | :--- |
| **Naive** | $640,703 | - |
| **Holt-Winters** | $529,096 | $111,607 (17.4%) |
| **LightGBM** | **$515,513** | **$125,190 (19.5%)** |
| **LSTM** | $1,229,646 | -91% (Loss) |

### 3.3 The Deep Learning "Failure" Case
Why did LSTM lose money?
-   **The Mean Bias:** LSTMs trained on MSE predict the "conditional mean" (e.g., 0.2 units) for sparse items.
-   **Inventory Consequence:** Predicting 0.2 when demand is 0 leads to overstock. Predicting 0.2 when demand is 5 leads to massive stockouts.
-   **Lesson:** For sparse data, "point prediction" Deep Learning models need probabilistic loss functions (like Negative Binomial Likelihood) to succeed.

---

## 4. Conclusion & Recommendations

1.  **For Production:** Deploy **LightGBM**. It offers the best ROI ($125k savings) and is computationally efficient.
2.  **For Interpretability:** Use **Holt-Winters**. It is nearly as accurate but easier to explain to stakeholders.
3.  **Future Work:** Investigate probabilistic Deep Learning (DeepAR) to solve the sparsity issue encountered by the standard LSTM.

---

## 5. References
1.  Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M5 Accuracy competition: Results, findings and conclusions.*
2.  Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice.*
