# **🧪 Algorithm Comparison Plan (10+ Models)**

To demonstrate rigor, we will benchmark models across three tiers of complexity.

## **🟢 Tier 1: "The Excel/Minitab Squad" (Easy & Fast)**

Responsibility: Team (Non-Coders)  
Tool: Excel or Minitab  
Dataset: The 50-item forecast\_sample\_for\_excel.csv (Historical data provided)

1. **Naive Method:** Forecast \= Last observed value. (The baseline to beat).  
2. **Simple Moving Average (SMA):** Average of last 28 days.  
3. **Weighted Moving Average (WMA):** Weights \[0.5, 0.3, 0.2\] on last 3 weeks.  
4. **Single Exponential Smoothing (SES):** $\\alpha \= 0.2$. Good for flat trends.  
5. **Holt’s Linear Trend:** Adds a trend component ($\\beta$). Good if sales are growing.  
6. **Holt-Winters (Additive):** Adds seasonality ($\\gamma$). The "Gold Standard" of classical methods.  
* **Deliverable:** A table in Excel with RMSE for these 6 methods on the 50 items.

## **🟡 Tier 2: "The Machine Learning Middle" (Medium)**

Responsibility: You (Dev) or Tech-Savvy Teammate  
Tool: Python (scikit-learn, statsmodels, prophet)  
Dataset: Full Dataset (or large sample)

7. **ARIMA / SARIMA:** The statistical heavyweight. Hard to tune manually, but pmdarima in Python automates it.  
8. **Facebook Prophet:** Excellent for handling holidays and weekly seasonality automatically. Very popular in industry.  
9. **XGBoost:** Similar to LightGBM but often behaves slightly differently. Good for comparison.  
10. **Random Forest Regressor:** Good baseline for tree-based methods. Slow on big data, but interpretable.

## **🔴 Tier 3: "The Deep Learning Heavyweights" (Hard & Slow)**

Responsibility: You (Dev)  
Tool: PyTorch / Python  
Dataset: Full Dataset

11. **LightGBM (Done):** Your current ML baseline. Fast and accurate.  
12. **LSTM (Done):** Your current Deep Learning champion. Handles sequences.  
13. **N-BEATS (Stretch):** A pure DL architecture specifically for time-series (beats M4 competition winners).  
14. **Temporal Fusion Transformer (TFT) (Stretch):** The state-of-the-art for interpretability \+ accuracy.

## **🚀 Execution Strategy**

### **Phase 1: The "Must-Haves" (Week 1\)**

* **Team:** Finish Tier 1 (Excel) on sample data.  
* **You:** Finish Optimization Script (optimize.py) using current LSTM results.

### **Phase 2: The "Nice-to-Haves" (Week 2-3)**

* **You:** Script prophet\_baseline.py (Tier 2\) and run it on the 50 items.  
* **You:** Script xgboost\_baseline.py (Tier 2\) \- easy copy/paste from your LightGBM script.

### **Phase 3: The "Flex" (End of Semester)**

* **You:** If time permits, try N-BEATS (Tier 3\) using the pytorch-forecasting library (wraps PyTorch comfortably).