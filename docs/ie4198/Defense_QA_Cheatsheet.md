# 🛡️ IE4198 Final Defense: The Ultimate Q&A Cheatsheet

This cheatsheet anticipates the hardest questions your jury might ask, especially the professor who knows Python well. It provides clear, confident, and highly technical answers.

---

## 1. 🐍 Data Engineering & Python (The "Polars" Questions)

**Q: Why did you switch from Pandas to Polars? Was it just for speed?**
> **A:** "It wasn't just for speed; it was for survival. The M5 dataset has 59 million rows. Pandas uses 'eager execution' and holds everything in RAM, which instantly triggered Out-Of-Memory (OOM) errors on our machines. Polars uses **Lazy Evaluation**—it builds a query plan first and only executes the necessary operations, reading directly from compressed Parquet files. It also utilizes multi-threading in Rust, which allowed us to engineer complex rolling features in seconds instead of hours."

**Q: You mentioned 'Parquet' files. Why not just use CSVs?**
> **A:** "CSVs are uncompressed text files that require the machine to infer data types on every read. Parquet is a columnar storage format. It strictly enforces data types (like `int16` or `float32`) and compresses the data massively. Loading a 59M row Parquet file takes a fraction of the RAM and time compared to a CSV."

**Q: How did you handle missing data or 'zeros' in the dataset?**
> **A:** "In retail, zeros aren't 'missing' data—they are true signals of zero demand. We did not impute or remove these zeros, as doing so would destroy the intermittent demand profile. Instead, we used algorithms like LightGBM and DeepAR that natively understand zero-inflated distributions."

---

## 2. 🧮 Methodology & Evaluation (The "Data Leakage" Questions)

**Q: How did you evaluate your models? Did you predict the unknown future?**
> **A:** "No. We used a strict temporal Hold-Out set. We trained our models on the first 1,885 days of data, and we used the final 28 days as our 'test set'. Our RMSE and financial simulations are calculated by comparing our models' predictions for those last 28 days directly against the actual, known sales."

**Q: Explain what 'Recursive Inference' is and why you needed it.**
> **A:** "If we want to predict 28 days into the future, we need features like `Lag_7` or `Rolling_Mean_28`. But on Day 15, we don't know the actual sales from Day 14 to calculate the lag! If we use the actual future sales, that's **data leakage**. Recursive Inference solves this by predicting Day 1, appending that prediction back into the dataset, dynamically recalculating the lags, and then predicting Day 2. It forces the model to live in reality."

**Q: Why did you use WRMSSE / RMSE instead of standard MAPE (Mean Absolute Percentage Error)?**
> **A:** "Because our dataset is highly intermittent. If a product sells 0 units, and the model predicts 1 unit, calculating a percentage error (MAPE) involves dividing by zero, which results in infinity. RMSE handles zeros perfectly and penalizes large errors heavily, which aligns with the high cost of massive stockouts or overstocks."

**Q: How did you ensure your models didn't just overfit the 5-year history?** *(NEW)*
> **A:** "Aside from the strict train/test split, algorithms like LightGBM and XGBoost have built-in L1 and L2 regularization penalties. This forces the decision trees to remain simple and stops them from memorizing noise. For DeepAR, we monitored the training loss against validation loss to implement early stopping."

---

## 3. 🤖 Algorithms (LightGBM vs DeepAR vs Classical)

**Q: Why did ARIMA and Holt-Winters fail on this dataset?**
> **A:** "Classical models assume a continuous, relatively smooth time series. They completely break down on 'intermittent' data where a product might sell 0 units for 5 days, then 10 units on a Saturday. They also cannot natively ingest external variables like SNAP benefit days or categorical IDs without extreme complexity (like SARIMAX), which doesn't scale to 30,000 items."

**Q: You mentioned LSTM had a 'Sparsity Penalty'. What does that mean?**
> **A:** "LSTM networks are great at finding patterns, but in our sparse data, the LSTM started predicting tiny decimal values (like 0.15 units) during dead periods instead of a hard zero. Mathematically, 0.15 gives a great RMSE. But operationally, a warehouse system sees 0.15 and might trigger a restock order. It slowly bled money in holding costs. We called this the Sparsity Penalty."

**Q: Why was LightGBM your 'Champion' model?**
> **A:** "LightGBM is a gradient-boosted decision tree. Decision trees handle zeros flawlessly—they simply create a branch saying 'if day is Tuesday and no promotion, predict 0'. It also handles categorical variables (like Store_ID) natively without needing One-Hot Encoding, which saved massive amounts of RAM."

**Q: DeepAR is a 'probabilistic' model. What does that actually mean?**
> **A:** "Unlike LightGBM which outputs a single number (a 'point forecast', e.g., 5 units), DeepAR outputs a mathematical distribution (e.g., a Negative Binomial distribution). This means DeepAR can tell us: 'I expect 5 units, but I am 90% confident it won't exceed 8 units.' This is incredibly powerful for calculating Safety Stock."

**Q: Did you use any external data outside of purely historical sales?** *(NEW)*
> **A:** "Yes! A purely autoregressive model fails during holidays. We heavily integrated external regressors, specifically calendar events (Super Bowl, Ramadan), SNAP food-stamp benefit days which drastically alter grocery purchasing velocities, and dynamic weekly sell prices."

---

## 4. 💰 Financial Simulation (Newsvendor Model)

**Q: How did you convert RMSE into a dollar amount?**
> **A:** "We used the Newsvendor Model. We took the exact predictions generated by the models and ran them through a simulated warehouse. If the model predicted 10, we stocked 10. If actual demand was 12, we incurred a 'Stockout Cost' for the 2 missed sales. If actual demand was 8, we incurred a 'Holding Cost' for the 2 unsold items. The Total Logistics Cost (TLC) is the sum of these penalties."

**Q: What is the 'Critical Ratio' in your dashboard?**
> **A:** "The Critical Ratio is the balance between the cost of holding inventory vs. the cost of a stockout. If holding a TV in a warehouse is cheap, but losing a TV sale is disastrous, the critical ratio is high. Our DSS dashboard lets managers adjust these costs dynamically, proving that the 'best' algorithm actually changes depending on the financial environment."

---

## 5. 🔍 Explainable AI (SHAP) & ESG Impact

**Q: What are SHAP values? Why not just use LightGBM's built-in 'Feature Importance'?**
> **A:** "Standard feature importance only tells us *how often* a feature was used to split a tree. It doesn't tell us the *direction* of the impact. SHAP (SHapley Additive exPlanations) uses Game Theory to calculate exactly how much a feature shifted the prediction. SHAP tells us not just that 'Price' is important, but that 'A high price decreased the forecast by exactly 2.5 units'."

**Q: How exactly does a forecasting algorithm have a Sustainability or ESG impact?** *(NEW)*
> **A:** "By shrinking safety stocks by 19.5%, we directly shrink the physical footprint required in the warehouse. Less physical inventory means lower electricity demands for HVAC and lighting. Furthermore, preventing stockouts drastically reduces the need for 'Rush Shipments', which typically rely on fast but highly-polluting transportation methods instead of efficient bulk freight."

---

## 6. 💻 Software Engineering (The Next.js DSS)

**Q: You built a Next.js dashboard. Does it run the AI models live in the browser?**
> **A:** "No, running a 30,000-SKU LightGBM model requires heavy compute. The models were trained offline in Python. We pre-calculated the predictions, actuals, and SHAP values, and the Next.js app ingests this data. The Next.js dashboard performs the *financial simulation* (the Newsvendor math) live in the browser, allowing instant, zero-latency sensitivity analysis for the managers."

**Q: Why use Next.js / React instead of just a Python Streamlit dashboard?**
> **A:** "While Streamlit is great for rapid prototyping, a Next.js React application represents a true, production-grade enterprise software architecture. It allows for infinite customizability in the UI/UX, better client-side state management for the interactive charts, and seamless deployment."
