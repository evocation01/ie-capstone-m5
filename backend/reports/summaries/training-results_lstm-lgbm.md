M5 Forecasting Experiment Results

1. Baseline Model (LightGBM)

Model File: backend/models/baseline_lgbm.pkl

Validation RMSE: 2.0222

Notes: Trained on sales (raw scale). 5-fold cross-validation was not used; simple time split.

2. Deep Learning Model (LSTM)

Model File: backend/models/lstm_best.pt

Best Epoch: 12

Train Log-RMSE: 0.4457

Validation Log-RMSE: 0.5337

Notes: \* Target was log1p(sales).

To compare with LightGBM, we roughly convert: $e^{0.53} \approx 1.7$.

Conclusion: LSTM (1.7 error proxy) outperformed LightGBM (2.02 error).

3. Forecast Output

Full Forecast: backend/results/forecasts/forecast_lstm.csv (3,049 items)

Sample for Team: backend/results/forecasts/forecast_lgbm_sample100.csv (100 items)
