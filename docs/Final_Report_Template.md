# IE Capstone Project: Final Report

**Project Title:** Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization  
**Course:** IE 4197 - Engineering Project I/II  
**Date:** January 2026  
**Team Members:** [Your Names Here]  
**Repository:** [Link to your GitHub Repo]

---

## 📄 Abstract
*(Approx. 250 words)*
Retail inventory optimization is a critical challenge characterized by sparse demand, high seasonality, and the need to balance holding costs against stockouts. This project investigates the efficacy of **10 forecasting algorithms**, ranging from classical statistical methods (Holt-Winters, ETS) to modern Machine Learning (LightGBM) and Deep Learning (LSTM), applied to the Walmart M5 dataset (30,490 time series).

Our benchmarking reveals that **LightGBM** (RMSE 1.41) and **Holt-Winters** (RMSE 1.45) significantly outperform standard Deep Learning approaches (LSTM RMSE 2.02) in this domain. By integrating these forecasts into a simulation of a standard `(s, Q)` inventory policy, we demonstrated that the LightGBM model reduces total logistics costs by **19.5% ($125,190)** compared to a naive baseline. The study concludes that for high-frequency, zero-inflated retail data, gradient boosting methods offer the optimal trade-off between accuracy, computational efficiency, and financial return.

---

## 1. Introduction

### 1.1 Project Background
Supply chain management relies fundamentally on accurate demand forecasting. In the retail sector, this is complicated by the "Bullwhip Effect," where small forecast errors amplify into massive inefficiencies upstream. The industry is currently shifting from traditional time-series methods to AI-driven approaches, but the financial benefit of this shift is often assumed rather than proven.

### 1.2 Problem Statement
Traditional metrics like RMSE do not always align with business objectives. A model might minimize squared error but fail to predict intermittent demand, leading to costly stockouts. This project addresses the gap between *forecast accuracy* and *inventory profitability*.

### 1.3 Objectives
1.  **Forecast:** Develop and benchmark models across three tiers: Classical, ML, and Deep Learning.
2.  **Optimize:** Simulate a theoretical inventory policy to translate accuracy into financial metrics (Holding Cost vs. Stockout Cost).
3.  **Compare:** Quantify the "Value of Information" provided by advanced AI models.

---

## 2. Methodology

### 2.1 Data Description (M5 Dataset)
We utilized the Kaggle M5 Forecasting dataset, covering 3,049 items across 10 Walmart stores (30,490 total time series) over a 5-year period (2011-2016).
*   **Sparsity:** ~50-70% of daily sales are zero.
*   **Hierarchy:** Item $\rightarrow$ Department $\rightarrow$ Category $\rightarrow$ Store $\rightarrow$ State.
*   **Features:** Calendar events (Super Bowl, Holidays), SNAP (Food Stamps), Sell Prices.

### 2.2 Model Architecture (The "Drag Race")
We implemented a rigorous benchmark of **10 models**:

| Tier | Category | Models Included | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **1** | **Classical** | Naive, SMA, WMA, SES, **Holt-Winters**, ETS, ARIMA | Statistical, interpretable, handles seasonality well. |
| **2** | **ML Baselines** | XGBoost, Random Forest, Prophet | Feature-based, non-linear relationships. |
| **3** | **Deep Learning** | **LightGBM**, LSTM | High capacity, sequence modeling (LSTM), tree-splitting (LGBM). |

### 2.3 Inventory Simulation
We simulated a continuous review `(s, Q)` policy over the 28-day validation period.
*   **Reorder Point ($r$):** Demand during lead time + Safety Stock.
*   **Safety Stock ($SS$):** $z \times \sigma_{error} \times \sqrt{L}$, where $\sigma_{error}$ is the model's RMSE.
*   **Cost Function:** $C_{total} = C_{holding} + C_{stockout}$

---

## 3. Results & Discussion

### 3.1 Forecast Accuracy (RMSE Leaderboard)
*Tested on 3,049 items over a 28-day horizon.*

| Rank | Model | RMSE | Insight |
| :--- | :--- | :--- | :--- |
| 🥇 | **LightGBM** | **1.4121** | Excellent handling of zero-inflated data via tree splits. |
| 🥈 | **Holt-Winters** | 1.4533 | Captured the strong weekly seasonality effectively. |
| 3 | SMA (28-day) | 1.4536 | Surprisingly robust; noise filtering helps. |
| ... | ... | ... | ... |
| 8 | LSTM | 2.0234 | Suffered from "conditional mean" bias on sparse data. |

### 3.2 Financial Impact Analysis
We translated the RMSE differences into financial outcomes.

| Model | Total Cost | Savings vs. Naive | ROI Analysis |
| :--- | :--- | :--- | :--- |
| **Naive** | $640,703 | - | Baseline cost. High stockouts. |
| **Holt-Winters** | $529,096 | **+17.4%** | Excellent ROI for low complexity. |
| **LightGBM** | **$515,513** | **+19.5%** | **Best performance.** Maximizes availability while minimizing waste. |
| **LSTM** | $1,229,646 | -91% | Failed to predict spikes, leading to catastrophic stockouts. |

### 3.3 The "Deep Learning Trap"
While LSTM is state-of-the-art for many tasks, it failed here due to the **Sparsity Penalty**. Trained on MSE, it converged to predicting small float values (e.g., 0.3) for intermittent items.
*   **Reality:** Demand is 0 or 1.
*   **Inventory Logic:** Rounding 0.3 often leads to under-ordering during spikes.
*   **Solution:** Future work should use probabilistic loss functions (e.g., Tweedie, Negative Binomial) instead of MSE.

---

## 4. Conclusion
1.  **Complexity $\neq$ Value:** The complex LSTM performed worse than the simple Holt-Winters.
2.  **The Winner:** **LightGBM** is the recommended model for deployment, saving **$125k** in the simulation.
3.  **Impact:** Improving forecast accuracy by ~15% (Naive vs LightGBM) yielded a ~20% reduction in logistics costs.

---

## 5. References

1.  **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2020). *The M5 Accuracy competition: Results, findings and conclusions.* International Journal of Forecasting.
2.  **Ke, G., et al.** (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NIPS).
3.  **Hyndman, R. J., & Athanasopoulos, G.** (2018). *Forecasting: principles and practice (2nd ed).* OTexts: Melbourne, Australia.
4.  **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory.* Neural Computation.
5.  **Salinas, D., et al.** (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks.* International Journal of Forecasting.

---

## 6. Appendix

### A.1 Key Algorithm Implementations
*Note: Full source code is available in the attached GitHub repository.*

**A.1.1 Classical Models (Holt-Winters)**
```python
def holt_winters(history, horizon=28):
    model = ExponentialSmoothing(
        history, 
        trend="add", 
        seasonal="add", 
        seasonal_periods=7
    ).fit()
    return model.forecast(horizon)
```

**A.1.2 Optimization Logic**
```python
def calculate_costs(forecast, actual, holding_cost, stockout_cost):
    diff = forecast - actual
    # Positive diff = Leftover Stock (Holding Cost)
    holding = np.maximum(diff, 0) * holding_cost
    # Negative diff = Missed Sales (Stockout Cost)
    stockout = np.maximum(-diff, 0) * stockout_cost
    return np.sum(holding), np.sum(stockout)
```

### A.2 Simulation Parameters
-   **Holding Cost:** $1.00 / unit / day
-   **Stockout Cost:** $10.00 / unit (Lost margin + Penalty)
-   **Lead Time:** 1 Day (Review Period)
-   **Service Level Target:** 95%

### A.3 Detailed Benchmark Results
*(Include the first 20 rows of `benchmark_results_full.csv` here or a screenshot of the table)*