# 🛒 Kaggle M5 Forecasting: Accuracy

## 📌 Project Overview

This dataset is derived from the **M5 Forecasting - Accuracy** competition hosted on Kaggle by the **Makridakis Open Forecasting Center (MOFC)** at the University of Nicosia, in partnership with **Walmart**.

### 🎯 The Challenge

The goal is to estimate the point forecasts of daily unit sales for **3,049 products** sold in **10 Walmart stores** located in **3 US States** (California, Texas, and Wisconsin) for a **28-day horizon**.

Unlike typical forecasting tasks, this challenge introduces complexity through:

-   **Hierarchical Data:** Sales can be aggregated by location (State > Store) or product (Category > Department > Item).
-   **Explanatory Variables:** Influence of price, promotions, day-of-the-week effects, and special events (Super Bowl, Ramadan, etc.).
-   **Intermittency:** Many items have sporadic sales (sparse data), making traditional error metrics potentially misleading.

---

## 📂 Dataset Structure

The dataset comprises **30,490 time series** (all combinations of 3,049 items × 10 stores). The historical range covers **1,913 days** (approx. 5 years), from **2011-01-29** to **2016-04-24**.

### 1. `sales_train_validation.csv` (The Core Data)

Contains the historical daily unit sales data per product and store.

-   **Dimensions:** 30,490 rows (Items) × 1,919 columns (Metadata + Days).
-   **Metadata Columns:** `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`.
-   **Time Series Columns:** `d_1` to `d_1913` (Daily unit sales).

### 2. `calendar.csv` (Time Features)

Contains information about the dates on which products are sold.

-   **date:** The actual calendar date (YYYY-MM-DD).
-   **wm_yr_wk:** The ID of the week the date belongs to.
-   **weekday / wday / month / year:** Temporal features.
-   **event_name_1 / event_type_1:** Special events (e.g., "SuperBowl", "Sport").
-   **snap_CA / snap_TX / snap_WI:** Binary flags (0/1) indicating if SNAP food stamps were allowed in that state on that date.

### 3. `sell_prices.csv` (Price History)

Contains information about the price of the products sold per store and date.

-   **store_id / item_id:** Foreign keys to link with sales data.
-   **wm_yr_wk:** Week ID (prices change weekly).
-   **sell_price:** The price of the item for that week/store.

### 4. `sample_submission.csv`

Defines the submission format for the competition.

-   Columns: `id` (validation & evaluation IDs) followed by `F1` through `F28` (the 28-day forecast).

---

## 🧠 Hierarchical Structure

The dataset supports aggregation at 12 levels:

1.  **Unit Sales:** All products, all stores (Global).
2.  **State:** CA, TX, WI.
3.  **Store:** CA_1, CA_2, ..., WI_3.
4.  **Category:** Hobbies, Household, Foods.
5.  **Department:** Hobbies_1, Hobbies_2, Household_1...
6.  **Item:** 3,049 unique products.
    ...and various combinations thereof.

## 🏢 Context: The Makridakis Competitions

The **M-Competitions**, organized by Spyros Makridakis, are the "Olympics" of the forecasting world. They aim to empirically compare forecasting methods to determine what actually works in practice, moving beyond theoretical properties.

-   **M4 (2018):** 100,000 time series.
-   **M5 (This Project):** The first M-competition held on Kaggle, emphasizing the synergy between traditional statistical methods (ARIMA, ES) and Machine Learning (Gradient Boosting, Deep Learning).

> _"Inaccurate business forecasts could result in actual or opportunity losses. This competition challenges you to use machine learning to improve forecast accuracy."_
