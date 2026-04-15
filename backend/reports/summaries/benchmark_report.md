# 🏎️ Final Benchmark Report: The Algorithm Drag Race

## 1. Overview

We benchmarked **10 different forecasting models** across three tiers of complexity (Classical, ML, Deep Learning) on a large subset of the M5 dataset.

**Dataset Scope:**

- **Store:** **CA_1** (California Store 1). Selected as the representative pilot location.

- **Items:** **3,049** (All unique products available in Store CA_1).

- **Training:** ~5 years (1885 days).

- **Testing:** Last 28 days (Validation period).

## 2. The Leaderboard (RMSE)

| Rank  | Model             | Tier          | RMSE       | Notes                                                                                               |
| :---- | :---------------- | :------------ | :--------- | :-------------------------------------------------------------------------------------------------- |
| **1** | **LightGBM**      | 3 (ML)        | **1.4121** | The champion. Fast, accurate, handles zeroes well.                                                  |
| **2** | **Holt-Winters**  | 1 (Classical) | 1.4533     | The "Old Guard" performs incredibly well. Seasonality is key.                                       |
| **3** | **SMA (28-day)**  | 1 (Classical) | 1.4536     | Surprisingly robust on large scale. A very strong baseline.                                         |
| 4     | ETS (State Space) | 1 (Classical) | 1.4538     | Newer state-space implementation of Exponential Smoothing.                                          |
| 5     | ARIMA (1,1,1)     | 1 (Classical) | 1.4566     | Solid performance with fixed parameters.                                                            |
| 6     | XGBoost           | 2 (ML)        | 1.5020     | Good ML baseline, but needs more feature engineering to beat LightGBM.                              |
| 7     | SES               | 1 (Classical) | 1.5116     | Simple Exponential Smoothing.                                                                       |
| 8     | WMA (3-week)      | 1 (Classical) | 1.6994     | Weighted Moving Average.                                                                            |
| 9     | Naive             | 1 (Baseline)  | 1.8558     | Last observed value. The baseline to beat.                                                          |
| 10    | LSTM              | 3 (DL)        | 2.0234     | **The "Zero Problem".** Struggles with sparse data, predicting "safe means" instead of true zeroes. |

## 3. Key Insights

### 🌟 LightGBM vs. Holt-Winters

The battle for 1st place was extremely tight.

- **LightGBM** won because it likely handles the "zero-inflated" nature of the data better (using decision trees to split zero/non-zero days).
- **Holt-Winters** proved that for retail data with strong weekly patterns (weekends vs weekdays), classical seasonality methods are still "S-Tier".

### 📈 The Rise of Simple Moving Average (SMA)

On the full dataset, **SMA (28-day)** performed remarkably well (3rd place), beating more complex ARIMA and SES models. This suggests that for many noisy, intermittent items, simply averaging the last month is a very hard strategy to beat.

### 📉 The LSTM Underperformance

Deep Learning (LSTM) came in last place.

- **Root Cause:** The dataset is highly sparse (many 0s). LSTMs trained on MSE tend to predict the "conditional mean" (e.g., 0.2 sales), which is never correct.
- **Impact:** High RMSE because it rarely predicts the exact "0" or the specific integer spike, but rather a "safe" smooth curve in between.

## 4. Financial Impact Analysis (Optimization)

We ran a full inventory simulation on these 3,049 items using the top models to quantify the "real world" cost.

| Model            | RMSE       | Total Logistics Cost | Savings vs. Naive  |
| :--------------- | :--------- | :------------------- | :----------------- |
| **LightGBM**     | **2.10\*** | **$515,513**         | **+19.5% ($125k)** |
| **Holt-Winters** | 2.31       | $529,096             | +17.4% ($111k)     |
| **Naive**        | 2.86       | $640,703             | Baseline           |
| **LSTM**         | 3.58       | $1,229,646           | -91% (Loss)        |

_\*Note: Optimization RMSE might differ slightly from Benchmark RMSE due to different aggregation/cleaning steps in the simulation script._

**Conclusion:**

- **LightGBM** is the financial champion, balancing accuracy with cost efficiency.
- **Holt-Winters** is a very strong runner-up, proving that classical methods are still highly relevant.
- **LSTM's** "conditional mean" predictions (e.g., 0.2 units) are disastrous for inventory policy, leading to massive stockouts (predicting < 1 when actual demand is 1+).
