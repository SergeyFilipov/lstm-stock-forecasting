# 📈 LSTM-Based Multi-Horizon Stock Forecasting

This project implements a deep learning pipeline for forecasting stock prices 10, 20, and 30 days into the future using an LSTM model with company-specific embeddings. The system supports S&P 500 stocks and includes full data preprocessing, model training, evaluation, Excel output, and interactive GUI components.

---
## 📄 Project Files

- `sp500_lstm_attention_forecaster.py` – full pipeline implementation
- `stock_data1.parquet` – raw input data (S&P 500)
- `forecast_results.xlsx` – forecast output
- `forecast_stds.pkl`, `training_history.pkl` – serialized diagnostics
- `*.png` – training and evaluation visualizations


## 🧩 1. Importing Libraries and Initial Setup

This block imports all necessary libraries for data processing, model building, evaluation, visualization, and GUI interaction.

- 📦 **Data manipulation**: `pandas`, `numpy`
- 📈 **Visualization**: `matplotlib`
- 🧠 **Machine learning**: `scikit-learn`, `tensorflow.keras`
- 💾 **File handling & serialization**: `os`, `pickle`
- 🖥️ **Graphical interface**: `tkinter`

The line `matplotlib.use("TkAgg")` ensures compatibility with `tkinter` windows when displaying training plots.

> 📎 Defined in: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py)


## 🧩 2. GUI: Parquet File Previewer

A simple Tkinter-based GUI is used to preview `.parquet` files before loading them into the model pipeline.

- 📂 Opens a file dialog for `.parquet` selection
- 🧾 Displays the content in a scrollable text window (without scientific notation)
- ✅ Confirms upload and shows preview in a new window

> 🎯 Useful for quick inspection of raw data before transformation  
> 📎 Function: `view_parquet_file()`  
> 🗂️ File: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L24-L60)


## 📊 3. Data Loading, Transformation & Normalization

This block loads the raw `.parquet` file and prepares it for modeling by transforming it from long to wide format, coercing numeric types, and applying per-company normalization.

### 🔄 Steps:

- 🗂 **Load & pivot data** from long to wide:  
  Converts rows by `Indicator` into separate columns (`Open`, `High`, `Low`, `Close`, `Volume`)
- 🧹 **Clean and convert**:
  - Coerces all price/volume columns to numeric
  - Drops rows with missing values
  - Sorts by company and date
- 🧠 **Encode companies** with `LabelEncoder` (as required for embedding layer)
- 📏 **MinMax Scaling (0–1)** per company for `Close` and `Volume`  
  – stored in a `scalers` dictionary for inverse transformation later

> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L63-L87)  
> 📁 Source data: [`stock_data1.parquet`](stock_data1.parquet)


## 🧮 4. Sequence Generation for LSTM Input

This section builds the supervised learning dataset by creating time series sequences per company. Each input contains 20 consecutive days of scaled `Close` and `Volume`, used to predict the closing price 10, 20, and 30 days ahead.

### 🛠 Structure of the data:

- 🗂 **Per company**, sliding window of 20 days
- 📈 **Target variables**:
  - `y10`: Close price 10 days ahead
  - `y20`: Close price 20 days ahead
  - `y30`: Close price 30 days ahead
- ⚙️ Converts lists into:
  - `X_price`: shape `(samples, 20, 2)`
  - `X_company`: shape `(samples, 1)` (for embedding)
  - `y10`, `y20`, `y30`: shape `(samples,)`

> ⚠️ Companies with missing scaled values are automatically skipped  
> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L89-L110)


## 🧠 5. Model Architecture, Training & Saving

This section handles the full modeling pipeline: dataset splitting, neural architecture creation (with embeddings), training, and checkpointing.

### 📊 Dataset Split:
- **Train**: 70%
- **Validation**: 20%
- **Holdout**: 10%

### 🧱 Model Structure:
- 🔢 `Company_Input`: one-hot encoded and passed through an `Embedding` layer (`dim=8`)
- 🔁 `Price_Input`: 20-day sequence of `Close` and `Volume`, passed through:
  - `LSTM(128)` with dropout
  - `LSTM(64)` with dropout
- 🧬 `Concatenate()`: joins time-series output with company embedding
- 🧠 Dense layers: `Dense(64)` → `Dropout` → `Dense(32)`
- 🎯 Multi-output: 3 separate heads for 10, 20, and 30-day price forecasts

### 🛠 Training Details:
- Loss: Mean Squared Error (MSE) per horizon
- Optimizer: Adam
- Epochs: 100
- Batch size: 64
- Callbacks:
  - `ModelCheckpoint`: saves the best model by `val_loss`
  - `EarlyStopping`: stops training after 8 stagnant epochs

✅ If a previously trained model exists, it is loaded automatically to skip retraining.

> 🧾 File: [`stock_lstm_model.keras`](stock_lstm_model.keras)  
> 🧪 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L113-L170)


## 📈 6. Model Predictions & Error Evaluation

After training, the model generates forecasts for all three data splits: training, validation, and holdout. Mean Squared Error (MSE) is calculated for each forecast horizon.

### 📊 Evaluation per Horizon:
- `10-day`, `20-day`, and `30-day` MSEs are computed for:
  - 🔹 Training set
  - 🔸 Validation set
  - 🔻 Holdout set (used only for final, unbiased evaluation)

### 📐 Error Standard Deviations:
- Residual standard deviations (`σ₁₀`, `σ₂₀`, `σ₃₀`) are calculated from the holdout set
- Used later to build **95% confidence intervals** around each prediction

### 💾 Output:
- All standard deviations are saved in: [`forecast_stds.pkl`](forecast_stds.pkl)
- Training loss (if available) is optionally loaded from: [`training_history.pkl`](training_history.pkl)

> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L172-L200)


## 📉 7. Training Visualization

If training history is available (`training_history.pkl`), two types of diagnostic plots are generated:

### 📊 A. Total Loss per Epoch
Tracks the overall Mean Squared Error (MSE) during training and validation.

![](loss_plot.png)

### 📊 B. MSE by Forecast Horizon
Plots separate loss curves for 10-day, 20-day, and 30-day predictions — useful for detecting horizon-specific overfitting or instability.

![](mse_by_horizon.png)

> 🖼️ Saved files:  
> - [`loss_plot.png`](loss_plot.png)  
> - [`mse_by_horizon.png`](mse_by_horizon.png)  
> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L202-L226)


## 🔁 8. Inverse Scaling & Confidence Intervals

Before generating final forecasts, the model performs:

### 🔄 A. Inverse Transformation of Predictions
Since data was MinMax-scaled per company, predicted values must be converted back to actual prices using only the `Close` column.

- Function: `inverse_close_only(scaler, scaled_close)`
- Restores original scale using the company-specific scaler

### 📏 B. Residual-Based Error Estimation
To construct 95% confidence intervals, residuals between predicted and actual values (on the validation set) are computed:

- `std10`, `std20`, `std30`: standard deviation of residuals for each horizon
- `z = 1.96` used for building the confidence band

> ⚠️ If validation predictions are unavailable, defaults to 0 and disables interval estimation  
> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L228-L243)


## 📤 9. Forecast Generation & Interactive GUI

The final block generates multi-horizon forecasts for selected companies, calculates growth rates, builds confidence intervals, and exports results to Excel — all through an interactive GUI.

---

### 🧠 `run_forecast()` Function

Generates a 10/20/30-day forecast for each selected company:

- 🔁 Uses the last 20-day sequence for each company
- 🔢 Predicts scaled values, then inversely transforms to actual prices
- 📏 Builds **95% confidence intervals** using residual standard deviations:
  - `±1.96 × σ`
- 📈 Calculates **% growth** from the current price
- 🧮 Computes **classic β coefficient**:
  - From log returns of the company vs. the market (mean of all companies)

> 📁 Output file: [`forecast_results.xlsx`](forecast_results.xlsx)

---

### 🖥️ GUI: Company Selection for Forecasting

A dynamic `tkinter` window allows users to:

- ✅ Select one or multiple companies from the list
- 🎛 Use `Select All` / `Clear All` buttons
- 🚀 Click `Run the forecast` to execute predictions

If no validation residuals are available, the forecast is skipped for that company with a warning.

> 📎 Code section: [`sp500_lstm_attention_forecaster.py`](sp500_lstm_attention_forecaster.py#L245-Lend)  
> 📎 Output: [`forecast_results.xlsx`](forecast_results.xlsx)

---

> ✅ This block marks the end-to-end functionality: from raw data → trained model → real-time forecast with intervals and beta analysis.


## 📊 10. Forecast Accuracy Evaluation: 10-Day Horizon

This section performs **quantitative evaluation and visual analysis** of the model’s 10-day forecasts, using real stock prices observed on **May 17, 2025**.

---

### 📁 Evaluation Input:
- Excel file: [`Forecast vs Real Stock Prices by Company (10,20,30-Day Horizons).xlsx`](Forecast%20vs%20Real%20Stock%20Prices%20by%20Company%20%2810%2C20%2C30-Day%20Horizons%29.xlsx)
- Columns renamed:
  - `"Estimated price in $ in 10 days"` → `Forecast_10d`
  - `"Real price in $ in 10 days"` → `Real_10d`

---

### 📐 Regression Metrics:

- **MAE**: 15.27 $
- **RMSE**: 26.44 $
- **R² Score**: 0.9897 (≈ 99.0%)
- **Mean Relative Error**: –8.61 %
- **n (Companies Evaluated)**: 366

---

### 📈 A. Normalized Line Plot: Forecast vs Real Price

This plot compares forecasted and actual prices after normalization to `[0, 1]`, helping assess the shape similarity across all companies.

![](01_forecast_vs_real_normalized.png)

```
MAE = 15.27 $  
RMSE = 26.44 $  
R² = 0.9897 (99.0 %)
```

---

### 📉 B. Histogram: Relative Error Distribution

This histogram shows the spread of relative forecast errors across all companies. Most predictions fall within a ±10% error margin, though the distribution is slightly left-skewed (underprediction tendency).

![](02_relative_error_distribution.png)

```
Mean Relative Error = –8.61 %
```

---

### 📊 C. Scatter Plot (Log-Log Scale)

A log-log scatter plot comparing real and predicted prices. The diagonal red dashed line represents perfect prediction. The high correlation (Pearson r = 0.9973) indicates strong linearity and model stability across scales.

![](03_forecast_vs_real_loglog.png)

```
Pearson r = 0.9973  
n = 366
```

---

> 📎 Code section: `sp500_lstm_attention_forecaster.py`  
> 🖼️ Outputs:
> - `01_forecast_vs_real_normalized.png`
> - `02_relative_error_distribution.png`
> - `03_forecast_vs_real_loglog.png`


---

## 📚 Citation

If you use this project in your research or applications, please cite:

**Filipov, S.** (2025). *LSTM-Based Multi-Horizon Forecasting for S&P 500*. GitHub Repository.  
Available at: [GitHub Repo Link]

