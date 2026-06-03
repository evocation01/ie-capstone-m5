# IE4198 Final Defense - Speaker Script & Transitions

This cheatsheet provides a structured flow for your presentation. It uses **Speaker A, B, and C** so you can easily divide the sections among the 3 or 4 attending members. 

> [!TIP]
> **Pro-Tip for Absences:** If a member cannot attend, simply have the remaining members absorb their assigned "Speaker" block. The script is modular by design.

---

### **Slide 1: Title Slide**
**[Speaker A]**
"Welcome, and thank you for being here. Today we are presenting our IE4198 Capstone Project: A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization. Our team consists of Hakan, Boran, Deniz, and Ali, and we are supervised by Prof. Dr. Serol Bulkan."

### **Slide 2: Agenda**
**[Speaker A]**
"Here is a brief look at our agenda. We’ll start with the business problem that triggered this research, dive into how we scaled our data infrastructure using Polars, walk you through our forecasting methodology, and finally demonstrate our interactive Next.js Decision Support System."

### **Slide 3: Problem Definition**
**[Speaker A]**
"We started with a massive industry challenge: the Bullwhip Effect. In modern retail, highly intermittent or 'zero-inflated' sales data absolutely breaks traditional forecasting methods. This leads to bloated warehouses or costly stockouts. Our goal was clear: bridge the gap between high-end AI precision and practical inventory economics like the (Q, r) continuous review model, ultimately minimizing Total Logistics Cost."

### **Slide 4: Transition to IE4198**
**[Speaker B]**
"In the first semester (IE4197), we established our theoretical baseline. But this semester, we engineered a completely global, production-ready system. We implemented Recursive Inference to prevent data leakage, scaled globally to 10 Walmart stores simultaneously, utilized Probabilistic Deep Learning via DeepAR, and integrated Game Theory SHAP values to explain the AI’s decisions."

### **Slide 5 & 6: Project Timeline**
**[Speaker B]**
"Our work plan spanned the entire academic year. As you can see, the Fall semester was dedicated to literature, methodology, and small-scale testing. The Spring semester was dedicated to global big-data scaling, advanced ensemble training, and full-stack software development for our interactive dashboard."

### **Slide 7: Big Data Strategy: Polars**
**[Speaker C]**
"To scale globally, we faced a '59 Million Row Challenge'. Standard Pandas triggered out-of-memory errors instantly. To solve this, we migrated our entire pipeline to Polars. By leveraging lazy evaluation and strict schema enforcement over Parquet files, we achieved sub-second feature engineering across the entire 6-year history of Walmart’s data."

### **Slide 8: Methodology: Recursive Inference**
**[Speaker C]**
"A major flaw in academic forecasting is data leakage—predicting 28 days ahead by accidentally looking at future lags. We engineered a strict Recursive Inference loop. Our models predict Day 1, append it to the dataset, dynamically recalculate rolling features like `Lag_7`, and only then predict Day 2. This guarantees absolute mathematical integrity."

### **Slide 9: Methodology: Tournament**
**[Speaker C]**
"We structured our algorithms into three tiers: 
1. **Classical methods** like ARIMA and Holt-Winters as our baseline.
2. **Machine Learning Ensembles** like LightGBM, which handles complex categorical data natively.
3. **Deep Learning architectures** like LSTM and DeepAR, which provide probabilistic distributions to assess risk."

### **Slide 10: Global Forecast Results**
**[Speaker A]**
"When evaluated globally across all 30,490 SKUs, the results were clear. Taking the advice from last semester, we ensured our models were rigorously validated: we trained our models strictly on the first 1,885 days and mathematically tested our predictions against the actual, known sales of the final 28 days. 

While classical models struggled with sparse demand during this 28-day test, our Machine Learning architectures captured the intermittent patterns efficiently, drastically lowering the global Root Mean Squared Error."

### **Slide 11: Inventory Optimization** *(The Transition Hook)*
**[Speaker A]**
"But a lower RMSE means nothing if it doesn’t save money. By feeding these forecasts into a (Q,r) inventory simulation, we found that under standard conditions, LightGBM trimmed logistics spending by over 26%, saving roughly $167,000 annually against the naive baseline. 

**[Transition Hook]:** *Under standard industry conditions, LightGBM is the undisputed champion. But in the real world, supply chain costs are never static. We asked ourselves: Does LightGBM always win? What if holding costs drop to zero, or stockout penalties skyrocket? To answer that, we didn’t just run static numbers—we built a dynamic application.*"

### **Slide 12: XAI: Explainability with SHAP**
**[Speaker B]**
"Before we show you that application, we needed to ensure managers could trust it. By using SHAP values, we cracked open the 'Black Box'. We proved that rolling 28-day means and 7-day lags were the absolute strongest drivers for accuracy, while external factors like SNAP benefit days heavily dictated food purchasing velocity."

### **Slide 13 & 14: Next.js DSS Dashboard & Live Demo**
**[Speaker B]**
"Which brings us to our final delivery: The Decision Support System. We built a full-stack Next.js application that operationalizes our Big Data. 

*(Switch to Live Demo)*
As you can see, when we drag the holding cost slider or adjust the critical ratio, the optimal algorithm actually changes in real-time. For instance, in extremely low-holding-cost scenarios, DeepAR actually overtakes LightGBM due to its conservative safety buffers. This interactive approach bridges the gap between data science and executive planning."

### **Slide 15: Sustainability & ESG**
**[Speaker C]**
"Finally, financial savings aren't the only benefit. Better forecasting has a massive ESG impact. By reducing stored inventory by nearly 20%, we simultaneously reduce warehouse HVAC energy demands, lower the carbon footprint of rush shipping, and drastically reduce food spoilage waste."

### **Slide 16 & 17: References & Conclusion**
**[Speaker A]**
"In conclusion, we successfully scaled a pilot study into a global, production-ready AI forecasting tool, backed by an interactive financial dashboard that proves tangible business value.

Thank you for listening. We would now be happy to answer any questions you may have."
