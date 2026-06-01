Slide 1: Title Slide
Title: A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization
Subtitle: Industrial Engineering Capstone Project | IE 4198
Presenters: Hakan İspir • Boran Turan • Deniz Yağmur Adaş • Ali Kahya
Supervisor: Prof. Dr. Serol BULKAN
Institution: Marmara University | Industrial Engineering Department

Slide 2: Table of Contents
- Problem Definition
- Transition to IE4198: Key Advancements
- Big Data Strategy: Polars ETL
- Methodology: Recursive Inference & Lags
- Methodology: Horizontal Benchmarking
- Forecast Accuracy Results
- Newsvendor Inventory Optimization
- XAI & Explainability (SHAP)
- Next.js Decision Support System
- Sustainability & ESG Impact
- Conclusion & Managerial Insights

Slide 3: Problem Definition & The Bullwhip Effect
- Demand Volatility: Modern retail relies on intermittent, "zero-inflated" sales patterns that break traditional forecasting.
- The Bullwhip Effect: Small checkout fluctuations ripple up the supply chain, creating massive inventory distortions.
- Financial Imbalance: The constant trade-off between excessive Holding Costs (warehouse waste) and Stockout Penalties (lost margins).
- Goal: Minimize "Total Logistics Cost" by aligning forecasting accuracy with real-world inventory economics.

Slide 4: Transition to IE4198 (Key Advancements)
Moving beyond the static sample of IE4197, we introduced 5 major advancements:
1. Recursive Inference: Eliminating data leakage via dynamic day-by-day lag updates.
2. Global Scaling: Training our models on all 10 Walmart stores and 30,490 SKUs simultaneously.
3. Horizontal Benchmarking: Integrating DeepAR to test probabilistic Deep Learning vs. LightGBM.
4. Game Theory XAI: Utilizing SHAP to crack the "black box" and expose feature influence.
5. Interactive DSS: Building a React/Next.js dashboard for real-time Newsvendor sensitivity analysis.

Slide 5: Big Data Strategy (Polars ETL)
- Challenge: 5 years of daily records for 30k SKUs melted down to ~59 Million rows. Standard Pandas operations trigger Out-Of-Memory (OOM) errors.
- Solution: Transitioned entirely to the Polars library.
- Memory-Efficient Pipelines: Lazy Evaluation & Schema Enforcement on Parquet files.
- Sub-Second Feature Engineering: Allowed us to quickly generate complex rolling averages (e.g., 28-day lags) across 60M rows.

Slide 6: Recursive Inference Engineering
- The "Data Leakage" Problem: Predicting 28 days into the future all at once creates look-ahead bias when using rolling features.
- Our Solution: A recursive inference loop.
- Step 1: Predict Day t+1.
- Step 2: Append the prediction to the dataset.
- Step 3: Recompute lags (e.g., lag_7) and rolling means dynamically.
- Step 4: Predict Day t+2 using mathematically sound features.

Slide 7: Forecasting Models (Horizontal Benchmarking)
- Classical (IE Baselines): Naive, Holt-Winters, ARIMA. (Robust for stable cycles, weak on covariates).
- ML Ensembles: LightGBM (Gradient Boosting). Extremely fast leaf-wise tree splitting; dominant on structured tabular data.
- Deep Learning (Probabilistic): DeepAR. Uses recurrent neural networks to output predictive distributions instead of point forecasts, naturally modeling uncertainty and intermittent zero-sales.

Slide 8: Forecast Accuracy Results (RMSE)
Global Evaluation across 10 Stores:
1. LightGBM: RMSE 1.96 (Champion)
2. DeepAR: RMSE 2.15
3. Moving Average: RMSE 2.26
4. Holt-Winters: RMSE 2.31
5. Naive: RMSE 2.86 (Baseline)
- Insight: LightGBM’s tree-splits handled zero-inflation significantly better on average, reducing overall error compared to standard baselines.

Slide 9: Financial Impact (Newsvendor Simulation)
Translating RMSE into Dollars via a (Q, r) Continuous Review Policy:
- Standard Economics (Holding $1.00 / Stockout $10.00):
  - LightGBM achieves ~$287k Total Cost (+14.7% over optimal), beating Naive's $337k.
- Extreme Intermittent Demand (Holding $0.10 / Stockout $10.00):
  - DeepAR Dominates: ~$272k Total Cost (+61.6% over optimal).
  - LightGBM struggles: ~$513k Total Cost (+27.4% over optimal).
- Why?: DeepAR's probabilistic bounds handle the heavy zero-inflation of cheap-to-store items significantly better than point-forecast models.

Slide 10: XAI & Explainability (SHAP Game Theory)
- The Black Box: Complex AI models (LightGBM) are traditionally uninterpretable.
- SHAP Integration: We utilized Shapley Additive exPlanations to assign a payout (importance) to every feature.
- Key Findings: 
  - `rolling_mean_28` and `lag_7` were the primary drivers of model decisions.
  - Price changes and specific Calendar events (SNAP days) had strong, non-linear impacts on the forecast.

Slide 11: Next.js Decision Support System Dashboard
- Operationalizing the Data: We built a fully functional web dashboard using Next.js, React, and Tailwind CSS.
- Features: 
  - Real-time manipulation of Holding and Stockout costs.
  - Dynamic re-calculation of the Critical Ratio (CR).
  - Visual Leaderboard shifting dynamically (e.g., DeepAR taking 1st place when Holding costs drop below $0.20).

Slide 12: Sustainability & ESG Impact
- Waste Reduction: Higher accuracy minimizes the physical waste and spoilage of expired products (FOODS category).
- Emissions Reduction: Minimizing "Rush Shipments" allows for high-efficiency scheduled logistics routing.
- Energy Footprint: -19.5% reduction in Total Logistics costs correlates directly with reduced warehouse energy demands (HVAC/Lighting) due to lower idle inventory levels.

Slide 13: Conclusion & Managerial Insights
- Algorithm Choice: LightGBM is the superior all-rounder for global retail. However, for cheap, highly intermittent items (CR < 0.2), DeepAR is strictly necessary.
- Tooling ROI: Moving from Pandas to Polars is a mandatory step for modern IE Big Data workflows.
- KPI Alignment: Industrial Engineering optimization must shift from minimizing arbitrary statistical errors (RMSE) to minimizing Total Logistics Cost.

Slide 14: Questions?
A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization
Hakan İspir • Boran Turan • Deniz Yağmur Adaş • Ali Kahya
Marmara University | Industrial Engineering Department
