import os
import pickle
import numpy as np
import matplotlib
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
from tkinter import ttk, messagebox
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tkinter import filedialog
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Embedding, Concatenate, Dropout, Flatten)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, "stock_data1.parquet")
model_path = os.path.join(script_dir, "stock_lstm_model.keras")

seq_length = 20


def view_parquet_file():
    file_path = filedialog.askopenfilename(filetypes=[("Parquet files", "*.parquet")])
    if not file_path:
        return

    try:
        df = pd.read_parquet(file_path)

        # Formatting: disable scientific notation
        pd.set_option('display.float_format', '{:.10f}'.format)

    except Exception as e:
        print(f"[ERROR] Failed to open file: {e}")
        return

    # Create a new preview window
    view_window = tk.Toplevel()
    view_window.title(f" Review of {os.path.basename(file_path)}")
    view_window.geometry("800x600")

    #  Create a text box with scrollbars
    text = tk.Text(view_window, wrap="none")
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_y = tk.Scrollbar(view_window, command=text.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    text.config(yscrollcommand=scrollbar_y.set)
    scrollbar_x = tk.Scrollbar(view_window, command=text.xview, orient=tk.HORIZONTAL)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    text.config(xscrollcommand=scrollbar_x.set)
    text.insert(tk.END, df.to_string(index=False))

    print("[INFO] File uploaded successfully.")


# Main window
root = tk.Tk()
root.title("Parquet Viewer")
root.geometry("300x100")
button = tk.Button(root, text="Open Parquet file", command=view_parquet_file)
button.pack(pady=20)
root.mainloop()

# === 1. LOADING AND PROCESSING DATA (long → wide) ===
df = pd.read_parquet(parquet_path)
df = df.pivot_table(index=["Date", "Company"], columns="Indicator", values="Value").reset_index()
df = df[["Date", "Company", "Open", "High", "Low", "Close", "Volume"]]

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Company", "Date"])

# === 2. ENCODING AND NORMALIZATION ===
label_encoder = LabelEncoder()
df["Company_ID"] = label_encoder.fit_transform(df["Company"])
scalers = {}
selected_features = ["Close", "Volume"]
df_scaled = df.copy()

for company_id in df["Company_ID"].unique():
    sub_idx = df[df["Company_ID"] == company_id].index
    scaler = MinMaxScaler()
    df_scaled.loc[sub_idx, [f + "_scaled" for f in selected_features]] = \
        scaler.fit_transform(df.loc[sub_idx, selected_features])
    scalers[company_id] = scaler

# === 3. SEQUENCES FOR LSTM ===
X_price, X_company, y10, y20, y30 = [], [], [], [], []

# We crawl all companies individually
for company_id in df["Company_ID"].unique():
    sub = df_scaled[df_scaled["Company_ID"] == company_id].reset_index(drop=True)
    for i in range(len(sub) - seq_length - 30):
        scaled_cols = [f + "_scaled" for f in selected_features]
        missing = [col for col in scaled_cols if col not in sub.columns]
        if missing:
            print(f"⚠️ Skipping — missing columns {missing} at a company {company_id}")
            continue

        feat_data = sub[scaled_cols].values
        X_price.append(feat_data[i:i + seq_length])
        X_company.append(company_id)  # adding the company ID to each row
        y10.append(sub["Close_scaled"].values[i + seq_length + 9])  # 10 days ahead
        y20.append(sub["Close_scaled"].values[i + seq_length + 19])  # 20 days ahead
        y30.append(sub["Close_scaled"].values[i + seq_length + 29])  # 30 days ahead

# Converting lists into NumPy arrays of appropriate shape
X_price = np.array(X_price).reshape(-1, seq_length, len(selected_features))  # (samples, timesteps, features)
X_company = np.array(X_company).reshape(-1, 1)  # (samples, 1)
y10 = np.array(y10)
y20 = np.array(y20)
y30 = np.array(y30)

# === 4. MODEL: CHARGING OR TRAINING ===
# ✂️ Data splitting (always done before if/else)
n = len(X_price)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

X_train_price, X_train_company = X_price[:train_end], X_company[:train_end]
y10_train, y20_train, y30_train = y10[:train_end], y20[:train_end], y30[:train_end]

X_val_price, X_val_company = X_price[train_end:val_end], X_company[train_end:val_end]
y10_val, y20_val, y30_val = y10[train_end:val_end], y20[train_end:val_end], y30[train_end:val_end]

X_hold_price, X_hold_company = X_price[val_end:], X_company[val_end:]
y10_hold, y20_hold, y30_hold = y10[val_end:], y20[val_end:], y30[val_end:]

# === Loading or training the model ===
if os.path.exists(model_path):
    print("🔁 Loading saved model...")
    model = load_model(model_path)
else:
    print("🧠 I am training a new model....")

    # 🧱 Creating the architecture
    company_input = Input(shape=(1,), name="Company_Input")
    embed = Embedding(input_dim=len(label_encoder.classes_), output_dim=8)(company_input)
    embed = Flatten()(embed)
    embed = Dropout(0.3)(embed)

    price_input = Input(shape=(seq_length, 2), name="Price_Input")
    lstm = LSTM(128, return_sequences=True, dropout=0.3)(price_input)
    lstm = LSTM(64, return_sequences=False, dropout=0.3)(lstm)

    concat = Concatenate()([lstm, embed])
    fc1 = Dense(64, activation="relu")(concat)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(32, activation="relu")(fc1)

    out10 = Dense(1, name="out10")(fc2)
    out20 = Dense(1, name="out20")(fc2)
    out30 = Dense(1, name="out30")(fc2)

    model = Model(inputs=[price_input, company_input], outputs=[out10, out20, out30])
    model.compile(optimizer="adam", loss={"out10": "mse", "out20": "mse", "out30": "mse"})

    # 🚂 Training
    history = model.fit(
        [X_train_price, X_train_company],
        [y10_train, y20_train, y30_train],
        validation_data=([X_val_price, X_val_company], [y10_val, y20_val, y30_val]),
        epochs=100,
        batch_size=64,
        callbacks=[
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        ],
        verbose=1
    )
    model.save(model_path)

# === 5. PREDICTIONS OF THE TRAINED MODEL ===
# ➤ Prediction on training data (Train)
y10_pred_train, y20_pred_train, y30_pred_train = model.predict([X_train_price, X_train_company], verbose=0)
print("Train MSE (10 days):", mean_squared_error(y10_train, y10_pred_train))
print("Train MSE (20 days):", mean_squared_error(y20_train, y20_pred_train))
print("Train MSE (30 days):", mean_squared_error(y30_train, y30_pred_train))

# ➤ Prediction on validation data (Validation)
y10_pred_val, y20_pred_val, y30_pred_val = model.predict([X_val_price, X_val_company], verbose=0)
print("Validation MSE (10 days):", mean_squared_error(y10_val, y10_pred_val))
print("Validation MSE (20 days):", mean_squared_error(y20_val, y20_pred_val))
print("Validation MSE (30 days):", mean_squared_error(y30_val, y30_pred_val))

# ➤ Prediction on holdout data (untrained, used only for final verification)
pred10_hold, pred20_hold, pred30_hold = model.predict([X_hold_price, X_hold_company])
print("Holdout MSE (10 days):", mean_squared_error(y10_hold, pred10_hold))
print("Holdout MSE (20 days):", mean_squared_error(y20_hold, pred20_hold))
print("Holdout MSE (30 days):", mean_squared_error(y30_hold, pred30_hold))

# Calculate standard deviations of errors on holdout
std10 = np.std(y10_hold.reshape(-1) - pred10_hold.reshape(-1))
std20 = np.std(y20_hold.reshape(-1) - pred20_hold.reshape(-1))
std30 = np.std(y30_hold.reshape(-1) - pred30_hold.reshape(-1))

# Saving to a pickle file
with open("forecast_stds.pkl", "wb") as f:
    pickle.dump({"std10": std10, "std20": std20, "std30": std30}, f)

history_path = os.path.join(script_dir, "training_history.pkl")
history_dict = None

if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history_dict = pickle.load(f)


# === 6. VISUALIZATION OF TRAINING ===
if history_dict is not None:
    # 📉 Graph of total train/val loss by epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title("LSTM Грешка по време на обучение")
    plt.xlabel("Епоха")
    plt.ylabel("MSE (Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.show()

    # 📉 MSE graph for each horizon separately (10, 20, 30 days)
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict['out10_loss'], label='Train Loss (10 дни)')
    plt.plot(history_dict['val_out10_loss'], label='Val Loss (10 дни)')
    plt.plot(history_dict['out20_loss'], label='Train Loss (20 дни)')
    plt.plot(history_dict['val_out20_loss'], label='Val Loss (20 дни)')
    plt.plot(history_dict['out30_loss'], label='Train Loss (30 дни)')
    plt.plot(history_dict['val_out30_loss'], label='Val Loss (30 дни)')
    plt.title("MSE по хоризонти (10/20/30 дни)")
    plt.xlabel("Епоха")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mse_by_horizon.png", dpi=300)
    plt.show()

model.summary()


# === 7. HELPER FUNCTION FOR INVERSE TRANSFORMATION OF 'Close' ONLY BY ITS SCALER ===
def inverse_close_only(scaler, scaled_close):
    """Inverse transformation of only the Close column with a scaler trained with 2 features."""
    dummy = np.zeros((1, 2))  # два фичъра — 'Close' и 'Volume'
    dummy[0, 0] = scaled_close
    return scaler.inverse_transform(dummy)[0, 0]


# === 8. CALCULATING RESIDUALS AND STANDARD DEVIATIONS ===
try:
    residuals_10 = y10_val - y10_pred_val.flatten()
    residuals_20 = y20_val - y20_pred_val.flatten()
    residuals_30 = y30_val - y30_pred_val.flatten()

    std10 = np.std(residuals_10)
    std20 = np.std(residuals_20)
    std30 = np.std(residuals_30)

    z = 1.96  # 95% confidence interval
except NameError:
    print("⚠️ Residuals calculation failed: validation predictions missing.")
    std10 = std20 = std30 = 0
    z = 0


# === 9. FINAL FORECASTS AND OUTCOME ===
def run_forecast(selected_companies):
    results = []
    for company_id in selected_companies:
        sub = df_scaled[df_scaled["Company_ID"] == company_id].sort_values("Date")
        if len(sub) < seq_length + 30:
            continue
        if std10 == 0 or std20 == 0 or std30 == 0:
            print(f"⚠️ Skipping {label_encoder.inverse_transform([company_id])[0]} — reliable intervals are missing.")
            continue

        feature_cols = [f + "_scaled" for f in selected_features]
        last_seq = sub[feature_cols].values[-seq_length:]
        X_pred_price = last_seq.reshape(1, seq_length, len(selected_features))
        X_pred_company = np.array([[company_id]])

        pred10, pred20, pred30 = model.predict([X_pred_price, X_pred_company], verbose=0)

        scaler = scalers[company_id]
        current_scaled_close = sub["Close_scaled"].values[-1]
        current_price = inverse_close_only(scaler, current_scaled_close)
        price10 = inverse_close_only(scaler, pred10[0][0])
        price20 = inverse_close_only(scaler, pred20[0][0])
        price30 = inverse_close_only(scaler, pred30[0][0])

        low10, high10 = sorted([
            inverse_close_only(scaler, pred10[0][0] - z * std10),
            inverse_close_only(scaler, pred10[0][0] + z * std10)
        ])

        low20, high20 = sorted([
            inverse_close_only(scaler, pred20[0][0] - z * std20),
            inverse_close_only(scaler, pred20[0][0] + z * std20)
        ])

        low30, high30 = sorted([
            inverse_close_only(scaler, pred30[0][0] - z * std30),
            inverse_close_only(scaler, pred30[0][0] + z * std30)
        ])

        beta_classic = None
        sub_ret = sub[["Date", "Close"]].copy()
        sub_ret["LogReturn"] = np.log(sub_ret["Close"]).diff()

        market_ret = df.groupby("Date")["Close"].mean()
        market_ret = np.log(market_ret).diff()
        sub_ret = sub_ret.merge(market_ret.rename("MarketReturn"), left_on="Date", right_index=True, how="left")
        sub_ret = sub_ret.dropna()

        if len(sub_ret) >= 2:
            cov = np.cov(sub_ret["LogReturn"], sub_ret["MarketReturn"])[0][1]
            var = np.var(sub_ret["MarketReturn"])
            beta_classic = cov / var if var != 0 else None

        # Calculating percentage growth from the current price
        growth10 = 100 * (price10 - current_price) / current_price
        growth20 = 100 * (price20 - current_price) / current_price
        growth30 = 100 * (price30 - current_price) / current_price

        results.append([
            label_encoder.inverse_transform([company_id])[0],
            round(current_price, 4),
            round(price10, 4), round(low10, 2), round(high10, 2), round(growth10, 2),
            round(price20, 4), round(low20, 2), round(high20, 2), round(growth20, 2),
            round(price30, 4), round(low30, 2), round(high30, 2), round(growth30, 2),
            float(round(beta_classic.item(), 4)) if beta_classic is not None else None
        ])

    columns = [
        "Company",
        "Price on 2025-05-07",
        "Forecast Price on 2025-05-17",
        "10d Forecast Low",
        "10d Forecast High",
        "10d Forecast Growth (%)",
        "Forecast Price on 2025-05-27",
        "20d Forecast Low",
        "20d Forecast High",
        "20d Forecast Growth (%)",
        "Forecast Price on 2025-06-06",
        "30d Forecast Low",
        "30d Forecast High",
        "30d Forecast Growth (%)",
        "Beta (Classic)"
    ]

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_excel("forecast_results.xlsx", index=False)
    messagebox.showinfo("Done", "The forecasts are recorded in forecast_results.xlsx")


# GUI
root = tk.Tk()
root.title("Selection of companies for forecasting")
root.geometry("450x500")

# Scrollable Frame
canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Dynamically creating checkbuttons
company_vars = {}
for company in label_encoder.classes_:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(scrollable_frame, text=company, variable=var, anchor="w")
    chk.pack(fill="x", padx=5, pady=2)
    company_vars[company] = var


# Buttons: Select all / Clear all
def select_all():
    for var in company_vars.values():
        var.set(True)


def deselect_all():
    for var in company_vars.values():
        var.set(False)


ttk.Button(root, text="Select all", command=select_all).pack(side="left", padx=5, pady=5)
ttk.Button(root, text="Clear them all", command=deselect_all).pack(side="left", padx=5, pady=5)


def on_forecast():  # Prediction button
    selected = [label_encoder.transform([comp])[0]
                for comp, var in company_vars.items() if var.get()]
    if not selected:
        messagebox.showwarning("Caution", "Please select at least one company.")
        return
    run_forecast(selected)


ttk.Button(root, text="Run the forecast", command=on_forecast).pack(side="right", padx=10, pady=5)
root.mainloop()


#  === 10. Forecast Evaluation and Visualization for 10-Day Horizon ===
file_path = os.path.join(script_dir, "Forecast vs Real Stock Prices by Company (10,20,30-Day Horizons).xlsx")

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
output_dir = os.path.dirname(file_path)
save = lambda fname: plt.savefig(os.path.join(output_dir, fname), dpi=300)

df = df.rename(columns={
    'Estimated price in $ in 10 days 05/17/2025': 'Forecast_10d',
    'Real price in $ in 10 days 05/17/2025': 'Real_10d'
})

df['Error'] = df['Forecast_10d'] - df['Real_10d']
df['AbsError'] = df['Error'].abs()
df['RelativeError'] = 100 * df['Error'] / df['Real_10d']

mae = mean_absolute_error(df['Real_10d'], df['Forecast_10d'])
rmse = mean_squared_error(df['Real_10d'], df['Forecast_10d'])**0.5
r2 = r2_score(df['Real_10d'], df['Forecast_10d'])
mean_rel = df['RelativeError'].mean()
top5 = df.sort_values('AbsError', ascending=False).head()

# === LINE GRAPH (NORMALIZED) ===
df_norm = df.copy()
max_price = df[['Forecast_10d', 'Real_10d']].max().max()
df_norm['Forecast_10d_norm'] = df['Forecast_10d'] / max_price
df_norm['Real_10d_norm'] = df['Real_10d'] / max_price

plt.figure(figsize=(12, 6))
plt.plot(df_norm['Forecast_10d_norm'], label='Forecast (normalized)', linewidth=2)
plt.plot(df_norm['Real_10d_norm'], label='Real (normalized)', linewidth=2)
plt.xlabel("Company Index")
plt.ylabel("Normalized Price")
plt.title("Forecast vs Real Price on 17.05.2025 (Normalized)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
textstr = f"MAE = {mae:.2f} $\nRMSE = {rmse:.2f} $\nR² = {r2:.4f} ({r2*100:.1f}%)"
plt.gcf().text(0.75, 0.75, textstr, fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))
plt.tight_layout()
save("01_forecast_vs_real_normalized.png")
plt.show()

# === HISTOGRAM OF RELATIVE ERRORS ===
df_filtered = df[df['RelativeError'] > -50]  # Премахване на outliers
plt.figure(figsize=(10, 5))
plt.hist(df_filtered['RelativeError'], bins=30, color='mediumseagreen', edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of Relative Forecast Errors (%)")
plt.xlabel("Relative Error (%)")
plt.ylabel("Number of Companies")
plt.grid(True, linestyle='--', alpha=0.6)
rel_stats = f"Mean Relative Error = {mean_rel:.2f}%"
plt.gcf().text(0.72, 0.75, rel_stats, fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))
plt.tight_layout()
save("02_relative_error_distribution.png")
plt.show()

# === SCATTER (LOG-LOG) ===
df_log = df[(df['Forecast_10d'] > 0) & (df['Real_10d'] > 0)]
plt.figure(figsize=(8, 8))
plt.scatter(df_log['Real_10d'], df_log['Forecast_10d'], alpha=0.6, color='steelblue', edgecolor='k', linewidth=0.5)
plt.plot([df_log['Real_10d'].min(), df_log['Real_10d'].max()],
         [df_log['Real_10d'].min(), df_log['Real_10d'].max()],
         'r--', label='Ideal Prediction (y = x)')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Real Price ($, log scale)")
plt.ylabel("Forecast Price ($, log scale)")
plt.title("Forecast vs Real Price (10d ahead, Log-Log Scale)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
pearson_r = np.corrcoef(df_log['Real_10d'], df_log['Forecast_10d'])[0, 1]
scatter_stats = f"Pearson r = {pearson_r:.4f}\nn = {len(df_log)}"
plt.gcf().text(0.65, 0.1, scatter_stats, fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))
plt.tight_layout()
save("03_forecast_vs_real_loglog.png")
plt.show()
