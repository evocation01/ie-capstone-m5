# 🏎️ Final Benchmark Report: The Algorithm Drag Race

## 1. Overview

We benchmarked **13 different forecasting models** across three tiers of complexity (Classical, ML, Deep Learning) on a stratified sample of 10 items from the M5 dataset. The goal was to identify the most accurate model (lowest RMSE) for this specific retail context.

**Dataset:**

-   **Items:** 10 (Stratified sample covering all departments: FOODS, HOBBIES, HOUSEHOLD).
-   **Training:** ~5 years (1885 days).
-   **Testing:** Last 28 days (Validation period).

## 2. The Leaderboard (RMSE)

| Rank     | Model            | Tier          | RMSE       | Notes                                                                                              |
| :------- | :--------------- | :------------ | :--------- | :------------------------------------------------------------------------------------------------- |
| 🥇 **1** | **LightGBM**     | 3 (ML)        | **0.7104** | The champion. Fast, accurate, handles zeroes well.                                                 |
| 🥈 **2** | **Holt-Winters** | 1 (Classical) | 0.7126     | The "Old Guard" performs incredibly well. Seasonality is key.                                      |
| 🥉 **3** | **SES**          | 1 (Classical) | 0.7130     | Simple Exponential Smoothing. Effective for flat trends.                                           |
| 4        | Prophet          | 2 (ML)        | 0.7139     | Very close to top 3. Good interpretability.                                                        |
| 5        | ARIMA (1,1,1)    | 1 (Classical) | 0.7162     | Solid, even with fixed parameters.                                                                 |
| 6        | Holt Linear      | 1 (Classical) | 0.7180     | Trend-only smoothing.                                                                              |
| 7        | XGBoost          | 2 (ML)        | 0.7231     | Competitive, but slightly behind LightGBM.                                                         |
| 8        | Random Forest    | 2 (ML)        | 0.7258     | Decent baseline for tree ensembles.                                                                |
| 9        | SMA (28-day)     | 1 (Classical) | 0.7272     | Simple Moving Average. Reliable but lags changes.                                                  |
| 10       | Auto ARIMA       | 2 (Classical) | 0.7378     | Surprising underperformance. Might be overfitting or getting stuck in local minima for some items. |
| 11       | WMA (3-week)     | 1 (Classical) | 0.7847     | Weighted Moving Average. Weights might need tuning.                                                |
| 12       | LSTM             | 3 (DL)        | 0.8864     | **The "Zero Problem".** Struggles with sparse data (see Analysis).                                 |
| 13       | Naive            | 1 (Baseline)  | 1.0023     | Last observed value. The baseline to beat.                                                         |

## 3. Key Insights

### 🌟 LightGBM vs. Holt-Winters

The battle for 1st place was extremely tight (RMSE 0.710 vs 0.712).

-   **LightGBM** won because it likely handles the "zero-inflated" nature of the data better (using decision trees to split zero/non-zero days).
-   **Holt-Winters** proved that for retail data with strong weekly patterns (weekends vs weekdays), classical seasonality methods are still "S-Tier".

### 📉 The LSTM Underperformance

Deep Learning (LSTM) came in 12th place, beating only the Naive baseline.

-   **Root Cause:** The dataset is highly sparse (many 0s). LSTMs trained on MSE tend to predict the "conditional mean" (e.g., 0.2 sales), which is never correct (you sell 0 or 1, not 0.2).
-   **Impact:** High RMSE because it rarely predicts the exact "0" or the specific integer spike, but rather a "safe" smooth curve in between.

### 📉 Auto ARIMA

Auto ARIMA performing worse than fixed ARIMA (1,1,1) is a classic example of **overfitting**.

-   The automated process minimizes AIC on the training set, which can lead to complex models (e.g., ARIMA(3,1,3)) that fail to generalize to the test set as well as a simpler, more robust ARIMA(1,1,1).

## 4. Recommendations for Optimization

Based on these results, the `optimize.py` script should focus on comparing the financial impact of:

1.  **LightGBM** (The ML Champion)
2.  **Holt-Winters** (The Classical Champion)
3.  **LSTM** (To demonstrate the "financial cost" of the sparsity problem)
4.  **Naive** (Baseline)

## 5. Financial Impact Analysis (Optimization)

We ran a full inventory simulation on 3,049 items using the top models to quantify the "real world" cost.

| Model | RMSE | Total Logistics Cost | Savings vs. Naive |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **2.10** | **$515,513** | **+19.5% ($125k)** |
| **Holt-Winters** | 2.31 | $529,096 | +17.4% ($111k) |
| **Naive** | 2.86 | $640,703 | Baseline |
| **LSTM** | 3.58 | $1,229,646 | -91% (Loss) |

**Conclusion:**
- **LightGBM** is the financial champion, balancing accuracy with cost efficiency.
- **Holt-Winters** is a very strong runner-up, proving that classical methods are still highly relevant.
- **LSTM's** "conditional mean" predictions (e.g., 0.2 units) are disastrous for inventory policy, leading to massive stockouts (predicting < 1 when actual demand is 1+).

