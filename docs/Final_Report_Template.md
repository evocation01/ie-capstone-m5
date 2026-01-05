[OUTER COVER - WHITE CARDBOARD]

<div align="center">

**MARMARA UNIVERSITY**  
**FACULTY OF ENGINEERING**

<br><br><br>

**A COMPARATIVE ANALYSIS OF CLASSICAL AND DEEP LEARNING FORECASTING FOR SUPPLY CHAIN INVENTORY OPTIMIZATION**

<br><br>

**Student Names:**  
150322052 - Hakan İspir  
150320024 - Boran Turan  
150319053 - Deniz Yağmur Adaş  
150320052 - Ali Kahya

<br><br><br><br>

**IE 4197 ENGINEERING PROJECT**  
**FINAL REPORT**

<br><br>

**Department of Industrial Engineering**

<br>

**Supervisor**  
**Prof. Dr. Serol Bulkan**

<br><br>

**ISTANBUL, 2026**

</div>

[PAGE BREAK]

[INNER COVER - ACCEPTANCE PAGE]

<div align="center">

**MARMARA UNIVERSITY**  
**FACULTY OF ENGINEERING**

<br>

**A COMPARATIVE ANALYSIS OF CLASSICAL AND DEEP LEARNING FORECASTING FOR SUPPLY CHAIN INVENTORY OPTIMIZATION**

<br>

**Student Names:**  
Hakan İspir  
Boran Turan  
Deniz Yağmur Adaş  
Ali Kahya

<br><br>

**IE 4197 ENGINEERING PROJECT**  
**FINAL REPORT**

<br><br>

**Department of Industrial Engineering**

<br>

**Supervisor**  
**Prof. Dr. Serol Bulkan**

<br><br>

**The project report has been approved by the following jury members.**

</div>

<br>

**Supervisor:**  
Prof. Dr. Serol Bulkan  
**(Signature)** ...........................................................

<br>

**Jury Member:**  
[Name]  
**(Signature)** ...........................................................

<br>

**Jury Member:**  
[Name]  
**(Signature)** ...........................................................

<br><br>

**Approval Date:** ... / ... / 2026

<br>

<div align="center">
**ISTANBUL, 2026**
</div>

[PAGE BREAK]

# ACKNOWLEDGEMENTS

We would like to express our deepest gratitude to our supervisor, **Prof. Dr. Serol Bulkan**, for his invaluable guidance, continuous support, and expert insights into Optimization Theory throughout this project. His mentorship was instrumental in defining the scope of this study and guiding the methodological framework for inventory policy design.

We also thank the **Marmara University Industrial Engineering Department** for providing the academic foundation necessary to undertake such a comprehensive study.

Finally, we acknowledge the open-source community and the **Makridakis Open Forecasting Center (MOFC)** for making the M5 Forecasting dataset available, which served as the backbone of our research.

<br>

**January, 2026**  
**The Project Team**

[PAGE BREAK]

# TABLE OF CONTENTS

| Section               | Title                                              | Page   |
| :-------------------- | :------------------------------------------------- | :----- |
| **ACKNOWLEDGEMENTS**  |                                                    | ii     |
| **TABLE OF CONTENTS** |                                                    | iii    |
| **ABSTRACT**          |                                                    | iv     |
| **ÖZET**              |                                                    | v      |
| **LIST OF SYMBOLS**   |                                                    | vi     |
| **ABBREVIATIONS**     |                                                    | vii    |
| **LIST OF FIGURES**   |                                                    | viii   |
| **LIST OF TABLES**    |                                                    | ix     |
| **1.**                | **INTRODUCTION**                                   | **1**  |
| 1.1.                  | Problem Definition                                 | 2      |
| 1.2.                  | Aim and Scope                                      | 3      |
| **2.**                | **RELATED LITERATURE**                             | **4**  |
| 2.1.                  | Time Series Forecasting in Retail                  | 4      |
| 2.2.                  | The Shift to Machine Learning                      | 5      |
| 2.3.                  | Inventory Control Theory                           | 6      |
| **3.**                | **METHODOLOGY**                                    | **8**  |
| 3.1.                  | Proposed System Architecture (Dual-Track Approach) | 8      |
| 3.2.                  | Mathematical Modeling of Inventory Policy          | 10     |
| 3.3.                  | Forecasting Algorithms Selected for Evaluation     | 12     |
| 3.4.                  | Data Processing Framework (Big Data Strategy)      | 14     |
| **4.**                | **RESULTS AND FINDINGS (PHASE I)**                 | **16** |
| 4.1.                  | Data Validation and Exploratory Analysis           | 16     |
| 4.2.                  | Large-Scale Algorithm Benchmarking (The Drag Race) | 18     |
| 4.3.                  | Financial Impact Analysis (Inventory Simulation)   | 20     |
| **5.**                | **PROJECT PLAN FOR IE 4198**                       | **21** |
| **6.**                | **CONCLUSION**                                     | **22** |
|                       | **REFERENCES**                                     | **23** |
|                       | **APPENDICES**                                     | **25** |

[PAGE BREAK]

# ABSTRACT

**A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization**

Inventory optimization in the modern retail sector is increasingly challenged by high product variety ("The Long Tail") and intermittent demand patterns, often leading to the "Bullwhip Effect." This project proposes a comprehensive framework to evaluate the financial impact of advanced forecasting methodologies—specifically comparing **10 forecasting algorithms** ranging from Classical Statistical methods (Holt-Winters, ETS) to Machine Learning (LightGBM) and Deep Learning (LSTM) architectures.

In the scope of IE 4197, the theoretical infrastructure and data processing pipelines have been established using the massive **M5 Walmart dataset (30,490 time series)**. A rigorous "Drag Race" benchmark was conducted on 3,049 items to identify the most viable candidates for the final optimization phase.

Our findings reveal a significant divergence between theoretical complexity and practical value. **LightGBM** (RMSE 1.41) and **Holt-Winters** (RMSE 1.45) significantly outperformed standard Deep Learning approaches (LSTM RMSE 2.02). By integrating these forecasts into a simulated $(s, Q)$ inventory policy, we demonstrated that the LightGBM model reduces total logistics costs by **19.5% ($125,190)** compared to a naive baseline.

The study concludes that for high-frequency, zero-inflated retail data, gradient boosting methods offer the optimal trade-off between accuracy, computational efficiency, and financial return. The subsequent phase (IE 4198) will focus on the deployment of a Decision Support System dashboard to visualize these strategic insights.

**Keywords:** Supply Chain Forecasting, Inventory Optimization, Deep Learning, LightGBM, Newsvendor Model.

[PAGE BREAK]

# ÖZET

**Tedarik Zinciri Envanter Optimizasyonu için Klasik ve Derin Öğrenme Tahminleme Yöntemlerinin Karşılaştırmalı Analizi**

Modern perakende sektöründe envanter optimizasyonu, yüksek ürün çeşitliliği ve kesikli talep yapıları nedeniyle giderek zorlaşmakta ve "Kamçı Etkisi"ne yol açmaktadır. Bu proje, Klasik İstatistiksel yöntemler (Holt-Winters, ETS) ile Makine Öğrenmesi (LightGBM) ve Derin Öğrenme (LSTM) mimarilerinin finansal etkilerini karşılaştıran kapsamlı bir çerçeve önermektedir.

IE 4197 kapsamında, büyük ölçekli **M5 Walmart veri seti (30.490 zaman serisi)** kullanılarak projenin teorik altyapısı ve veri işleme hatları (pipeline) kurulmuştur. Final optimizasyon aşaması için en uygun adayları belirlemek amacıyla 10 farklı algoritma üzerinde kapsamlı bir kıyaslama çalışması yapılmıştır.

Bulgularımız, teorik karmaşıklık ile pratik değer arasında önemli bir ayrışma olduğunu ortaya koymaktadır. **LightGBM** (RMSE 1.41) ve **Holt-Winters** (RMSE 1.45), standart Derin Öğrenme yaklaşımlarından (LSTM RMSE 2.02) önemli ölçüde daha iyi performans göstermiştir. Bu tahminlerin simüle edilmiş bir $(s, Q)$ envanter politikasına entegre edilmesiyle, LightGBM modelinin basit bir temel modele kıyasla toplam lojistik maliyetlerini **%19,5 (125.190 $)** oranında azalttığı gösterilmiştir.

Çalışma, yüksek frekanslı, sıfır ağırlıklı perakende verileri için gradyan artırma (gradient boosting) yöntemlerinin doğruluk, hesaplama verimliliği ve finansal getiri arasında en uygun dengeyi sunduğu sonucuna varmaktadır. Takip eden aşama (IE 4198), bu stratejik içgörüleri görselleştirmek için bir Karar Destek Sistemi panelinin devreye alınmasına odaklanacaktır.

**Anahtar kelimeler:** Tedarik Zinciri Tahminleme, Envanter Optimizasyonu, Derin Öğrenme, LightGBM, Gazeteci Çocuk Modeli.

[PAGE BREAK]

# LIST OF SYMBOLS

| Symbol       | Description                                    |
| :----------- | :--------------------------------------------- |
| $D_L$        | Demand during lead time                        |
| $L$          | Lead time (days)                               |
| $Q$          | Order quantity                                 |
| $r$          | Reorder point                                  |
| $SS$         | Safety stock                                   |
| $z_{\alpha}$ | Z-score associated with service level          |
| $\sigma_e$   | Standard deviation of forecast error (RMSE)    |
| $\alpha$     | Level smoothing factor (Exponential Smoothing) |
| $\beta$      | Trend smoothing factor                         |
| $\gamma$     | Seasonality smoothing factor                   |

# ABBREVIATIONS

| Abbreviation | Definition                                    |
| :----------- | :-------------------------------------------- |
| **AI**       | Artificial Intelligence                       |
| **ARIMA**    | Autoregressive Integrated Moving Average      |
| **DL**       | Deep Learning                                 |
| **EDA**      | Exploratory Data Analysis                     |
| **ETS**      | Error, Trend, Seasonality (State Space Model) |
| **IE**       | Industrial Engineering                        |
| **LGBM**     | Light Gradient Boosting Machine               |
| **LSTM**     | Long Short-Term Memory                        |
| **ML**       | Machine Learning                              |
| **RMSE**     | Root Mean Squared Error                       |
| **SKU**      | Stock Keeping Unit                            |
| **TLC**      | Total Logistics Cost                          |

[PAGE BREAK]

# LIST OF FIGURES

-   **Figure 3.1** The Dual-Track Methodology Process Flow ............................... 9
-   **Figure 3.2** Mathematical Framework for Inventory Optimization ..................... 11
-   **Figure 3.3** Proposed Data Pipeline Architecture (Polars & Python) ................. 14
-   **Figure 4.1** Zero-Inflation Analysis of the M5 Dataset ............................. 16
-   **Figure 4.2** Final Benchmark Leaderboard (RMSE Comparison) ......................... 18
-   **Figure 4.3** Visualization of the "Sparsity Penalty" in LSTM Predictions ........... 19
-   **Figure 4.4** Financial Impact Analysis (Cost Savings by Model) ..................... 20
-   **Figure 5.1** Gantt Chart for Project Phase II (IE 4198) ............................ 21

# LIST OF TABLES

-   **Table 2.1** Comparative Summary of Forecasting Methodologies in Literature ......... 5
-   **Table 3.1** Selected Algorithms for Benchmarking (10 Models) ....................... 13
-   **Table 4.1** Final RMSE Results from Large-Scale Benchmark (3,049 Items) ............ 19
-   **Table 4.2** Inventory Optimization Simulation Results (Cost Matrix) ................ 20

[PAGE BREAK]

# 1. INTRODUCTION

Efficient supply chain management relies heavily on the accuracy of demand forecasting. Inaccurate predictions lead to two opposing but equally damaging costs: holding costs from overstocking and opportunity costs from stockouts. The central problem addressed in this project is the optimization of these costs in a high-dimensional, sparse retail environment.

This project aims to bridge the gap between traditional Industrial Engineering approaches and modern Data Science techniques. By utilizing the M5 Forecasting dataset, covering 30,490 product time series, we aim to construct a rigorous comparison framework.

## 1.1. Problem Definition

The stochastic nature of consumer demand makes it impossible to predict sales with perfect accuracy. The problem is formulated as finding the forecasting method $M$ that minimizes the Total Logistics Cost function $C(M)$ under target Service Level constraints.

Specifically, we address the challenge of **intermittent demand** in large-scale retail, where traditional metrics like RMSE may not align with financial performance due to the zero-inflated nature of SKU-level sales data.

## 1.2. Aim and Scope

The primary aim of IE 4197 was to establish the methodological foundation, clean and validate the data, and select the appropriate algorithms for the final optimization phase.

The scope of this first phase includes:

-   **Data Architecture:** Designing a pipeline capable of processing 59 million rows using high-performance libraries (Polars).
-   **Theoretical Framework:** Defining the inventory policies based on Newsvendor logic and Safety Stock derivation.
-   **Benchmarking:** Conducting a rigorous "Drag Race" of 10 forecasting algorithms to determine the "State of the Art" for this specific dataset.

[PAGE BREAK]

# 2. RELATED LITERATURE

The literature review focuses on three converging domains: classical time-series analysis, machine learning applications in retail, and stochastic inventory control.

## 2.1. Time Series Forecasting in Retail

The foundation of retail forecasting lies in statistical analysis. Box and Jenkins (1970) established the ARIMA methodology, which models autocorrelations in stationary data. While effective for aggregated demand, ARIMA struggles with the high intermittency found in SKU-level data. Winters (1960) introduced the Holt-Winters Exponential Smoothing method to handle seasonality, which remains a standard industry benchmark due to its computational simplicity and interpretability.

## 2.2. The Shift to Machine Learning

In recent years, the focus has shifted to Machine Learning. The M5 Forecasting Competition (Makridakis et al., 2022) highlighted the dominance of Gradient Boosting Decision Trees (GBDT), specifically LightGBM (Ke et al., 2017). These models excel at handling heterogeneous features and "static" covariates (store ID, item category) that classical models cannot easily incorporate.

Deep Learning approaches, such as Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997), offer the ability to model long-term dependencies. However, applying LSTMs to sparse retail data remains challenging. Benidis et al. (2022) note that standard loss functions (MSE) can lead to models predicting "conditional means" rather than discrete zeros, a phenomenon we investigate as the "Sparsity Penalty."

## 2.3. Inventory Control Theory

Forecasting is merely an input to the ultimate goal: inventory control. Silver et al. (2016) provide the definitive framework for stochastic inventory management, defining the relationship between forecast error ($\sigma_e$) and Safety Stock ($SS$). Syntetos et al. (2016) argue that forecast accuracy metrics like RMSE are often disconnected from inventory performance, necessitating a simulation-based approach to evaluation.

[PAGE BREAK]

# 3. METHODOLOGY

This section details the theoretical and mathematical framework designed for the project.

## 3.1. Proposed System Architecture (Dual-Track Approach)

To ensure both statistical rigor and computational scalability, a "Dual-Track" methodology has been designed:

-   **Track A (Classical Baseline):** A stratified sampling strategy is employed to select representative SKUs. These are analyzed using Box-Jenkins (ARIMA) and Exponential Smoothing (ETS/HW) methodologies to establish a baseline for explainability and theoretical validation.
-   **Track B (AI Scalability):** A "Global Model" approach is designed using Gradient Boosting and Recurrent Neural Networks (RNNs). This track leverages the full dataset to capture cross-series correlations (e.g., how price changes in Store A affect Store B) that univariate methods miss.

## 3.2. Mathematical Modeling of Inventory Policy

The forecasted demand will be input into a continuous review $(s, Q)$ policy. The formulation is derived from Silver et al. (2016).

**Safety Stock Formulation:**
Safety stock ($SS$) serves as a buffer against forecast error variance ($\sigma_e^2$) and lead time ($L$) variability.

$$SS = z_{\alpha} \times \sigma_e \times \sqrt{L}$$

Where $z_{\alpha}$ represents the standard normal variate for the target Cycle Service Level (e.g., 95%, $z=1.645$).

**Total Logistics Cost Function:**
The objective function to be minimized is defined as:

$$Min \ TLC = (H \times SS) + (S \times E[US])$$

Where:

-   $H$: Holding cost per unit/year.
-   $S$: Stockout penalty cost per unit.
-   $E[US]$: Expected units short (dependent on forecast accuracy).

## 3.3. Forecasting Algorithms Selected for Evaluation

Based on the literature review, a diverse set of **10 algorithms** has been selected for the experimental design:

| Tier  | Category          | Algorithms                                                      | Characteristics                                                 |
| :---- | :---------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- |
| **1** | **Classical**     | Naive, SMA, WMA, SES, Holt-Linear, **Holt-Winters**, ETS, ARIMA | Statistical, interpretable, effective for seasonal data.        |
| **2** | **ML Baselines**  | XGBoost, Random Forest, Prophet                                 | Feature-based, non-linear relationships.                        |
| **3** | **Deep Learning** | **LightGBM**, LSTM                                              | High capacity, sequence modeling (LSTM), tree-splitting (LGBM). |

## 3.4. Data Processing Framework (Big Data Strategy)

Given the volume of data (59M rows), a specialized pipeline utilizing the **Polars** library was architected. This ensures efficient memory usage and allows for complex feature engineering (rolling windows, lag features) that are computationally prohibitive in standard tools like Excel or pandas.

-   **Ingestion:** Reading parquet files with schema enforcement.
-   **Feature Engineering:** Creating lag features ($t-7, t-28$) and rolling statistics.
-   **Validation:** Ensuring no data leakage between training (2011-2016) and validation sets.

[PAGE BREAK]

# 4. RESULTS AND FINDINGS (PHASE I)

This section presents the definitive findings from the benchmarking studies conducted in IE 4197.

## 4.1. Data Validation and Exploratory Analysis

Extensive EDA was performed to understand the sparsity of the M5 dataset. It was observed that approximately 60-70% of the daily sales entries are zero. This structural zero-inflation poses a specific challenge for MSE-based loss functions used in Deep Learning.

## 4.2. Large-Scale Algorithm Benchmarking (The Drag Race)

A comprehensive benchmark was conducted on **3,049 items** (a statistically significant subset representing all categories and stores) over a 28-day forecast horizon.

**Table 4.1: Final RMSE Results from Large-Scale Benchmark**

| Rank  | Model        | Tier          | RMSE       | Insight                                                         |
| :---- | :----------- | :------------ | :--------- | :-------------------------------------------------------------- |
| **1** | **LightGBM** | **ML**        | **1.4121** | **The Accuracy Champion.** Best at handling zero-inflated data. |
| 2     | Holt-Winters | Classical     | 1.4533     | Extremely competitive. Captures seasonality perfectly.          |
| 3     | SMA (28-day) | Classical     | 1.4536     | Simple Moving Average. surprisingly robust against noise.       |
| 4     | ETS          | Classical     | 1.4538     | State-space equivalent of Holt-Winters.                         |
| ...   | ...          | ...           | ...        | ...                                                             |
| 7     | Naive        | Baseline      | 1.8558     | The baseline to beat.                                           |
| 8     | LSTM         | Deep Learning | 2.0234     | **Underperformance:** Failed to handle sparsity effectively.    |

**Key Finding:** Complexity does not equal accuracy. The interpretable Holt-Winters model outperformed the complex LSTM, while the tree-based LightGBM offered the best overall performance.

## 4.3. Financial Impact Analysis (Inventory Simulation)

To translate these error metrics into business value, we ran a full inventory simulation using the costs parameters ($H=1$, $S=10$).

**Table 4.2: Inventory Optimization Simulation Results**

| Model            | Total Cost ($) | Holding Cost | Stockout Cost | Savings vs. Naive  |
| :--------------- | :------------- | :----------- | :------------ | :----------------- |
| **Naive**        | $640,703       | $55,783      | $584,920      | -                  |
| **Holt-Winters** | $529,096       | $47,364      | $481,732      | +17.4% ($111k)     |
| **LightGBM**     | **$515,513**   | **$46,119**  | **$469,394**  | **+19.5% ($125k)** |
| **LSTM**         | $1,229,646     | $296         | $1,229,350    | -91% (Loss)        |

**Conclusion:** Implementing **LightGBM** results in a **19.5% reduction in total logistics costs**, amounting to a theoretical saving of **$125,190** over just 28 days for the selected items.

[PAGE BREAK]

# 5. PROJECT PLAN FOR IE 4198

Having completed the infrastructure, benchmarking, and initial optimization in IE 4197, the plan for the final semester is as follows:

-   **Full-Scale Execution (Feb-Mar):** Train the finalized LightGBM and DeepAR (Probabilistic) models on the complete 30,490 item dataset.
-   **Advanced Optimization (Mar-Apr):** Refine the simulation to include variable lead times and dynamic pricing constraints.
-   **Financial Validation (Apr):** Perform sensitivity analysis on Holding vs. Stockout cost ratios.
-   **Dashboard Deployment (May):** Finalize the Next.js web application for interactive presentation of the "What-If" scenarios.

[PAGE BREAK]

# 6. CONCLUSION

In this engineering project (IE 4197), the foundation for a comprehensive comparative analysis of forecasting methods has been established. The problem of inventory optimization in sparse retail environments was defined, and a robust dual-track methodology was developed.

Our rigorous benchmarking of 10 algorithms proved that **LightGBM** is the superior candidate for this domain, offering a **19.5% cost reduction** compared to the baseline. We also identified a critical "Sparsity Penalty" in standard Deep Learning models, guiding our future research towards probabilistic forecasting methods for Phase II.

The project is on track to deliver a statistically rigorous and financially grounded solution in IE 4198.

[PAGE BREAK]

# REFERENCES

1.  **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2022). _The M5 Accuracy competition: Results, findings and conclusions._ International Journal of Forecasting.
2.  **Ke, G., et al.** (2017). _LightGBM: A Highly Efficient Gradient Boosting Decision Tree._ Advances in Neural Information Processing Systems (NIPS).
3.  **Hyndman, R. J., & Athanasopoulos, G.** (2018). _Forecasting: principles and practice (2nd ed)._ OTexts: Melbourne, Australia.
4.  **Hochreiter, S., & Schmidhuber, J.** (1997). _Long Short-Term Memory._ Neural Computation.
5.  **Silver, E. A., Pyke, D. F., & Thomas, D. J.** (2016). _Inventory and production management in supply chains._ CRC Press.
6.  **Box, G. E. P., & Jenkins, G. M.** (1970). _Time series analysis: forecasting and control._ Holden-Day.
7.  **Winters, P. R.** (1960). _Forecasting sales by exponentially weighted moving averages._ Management Science.

[PAGE BREAK]

# APPENDICES

## Appendix A: Key Algorithm Logic

**A.1 Classical Models (Holt-Winters Implementation)**

```python
def holt_winters(history, horizon=28):
    # Triple Exponential Smoothing with Additive Seasonality
    model = ExponentialSmoothing(
        history,
        trend="add",
        seasonal="add",
        seasonal_periods=7
    ).fit()
    return model.forecast(horizon)
```

**A.2 Optimization Cost Function**

```python
def calculate_costs(forecast, actual, holding_cost, stockout_cost):
    """
    Calculates financial costs (Holding + Stockout).
    """
    diff = forecast - actual
    # Positive diff = Leftover Stock (Holding Cost)
    holding = np.maximum(diff, 0) * holding_cost
    # Negative diff = Missed Sales (Stockout Cost)
    stockout = np.maximum(-diff, 0) * stockout_cost
    return np.sum(holding), np.sum(stockout)
```

## Appendix B: Simulation Parameters

-   **Holding Cost ($H$):** $1.00 / unit / day
-   **Stockout Cost ($S$):** $10.00 / unit (Lost margin + Penalty)
-   **Lead Time ($L$):** 1 Day (Review Period)
-   **Service Level Target:** 95% ($z = 1.645$)

```

```
