# **📊 Analysis of Forecasting & Optimization Results**

## **1\. Executive Summary**

This study compared three forecasting methodologies—Naive Baseline, Gradient Boosting (LightGBM), and Deep Learning (LSTM)—to determine their impact on Supply Chain Inventory Optimization using the M5 dataset.  
**Key Finding:** While Deep Learning (LSTM) successfully converged during training, the **LightGBM (Gradient Boosting)** model provided the superior balance of accuracy and cost-efficiency for this specific sparse retail dataset, reducing total logistics costs by **19.5%**.

| Model        | RMSE (Error) | Total Logistics Cost | Performance vs. Naive       |
| :----------- | :----------- | :------------------- | :-------------------------- |
| **Naive**    | 2.86         | $640,703             | Baseline                    |
| **LightGBM** | **2.10**     | **$515,513**         | **\+19.5% Savings ($125k)** |
| **LSTM**     | 3.58         | $1,229,646           | \-91% (Loss)                |

## **2\. Why did Deep Learning (LSTM) underperform?**

The LSTM model's significantly higher costs ($1.2M vs $515k) despite successful training convergence highlights a critical challenge in retail forecasting: **The Sparsity Penalty.**

### **A. The "Safe Mean" Problem**

-   **Data Reality:** The M5 dataset is highly sparse (approx. 90% of daily sales in our sample were 0).
-   **LSTM Behavior:** Trained on MSE loss in log-space, the LSTM learned to predict the "conditional mean" of the distribution. For a sparse item, this is often a small float like 0.3.
-   **Inventory Impact:**
    -   **Scenario 1 (Actual=0, Pred=0.3):** The system buys stock. Result: **Holding Cost** for no reason.
    -   **Scenario 2 (Actual=5, Pred=0.3):** The system buys almost nothing. Result: **Massive Stockout Cost**.
-   **Contrast:** LightGBM, using decision trees, could effectively "segment" the zero-sale days from the high-sale days, leading to a lower RMSE (2.10) and better inventory decisions.

### **B. Log-Space vs. Real-World Cost**

The LSTM minimized Log-RMSE effectively (0.53), but this non-linear transformation masks the impact of large errors in "real dollar" space. A small error in log-space can translate to a significant stockout penalty in the simulation.

## **3\. Conclusion & Recommendation**

For high-velocity, sparse retail data, **Gradient Boosting (LightGBM)** is the recommended production model. It offers:

1. **Lower Costs:** Saving $125,000+ compared to a Naive baseline.
2. **Robustness:** Better handling of zero-inflated data than standard LSTMs.
3. **Efficiency:** Faster training and inference times.

Future work to improve Deep Learning performance should focus on **Probabilistic Architectures** (e.g., DeepAR) that explicitly model the probability of "zero" versus "spike," rather than standard regression LSTMs.
