import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor
import os
from sklearn.model_selection import train_test_split

MODEL_DIR = "agModel"
DATA_PATH = "data/cleaned.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

predictor = TabularPredictor.load(MODEL_DIR, require_py_version_match=False)
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

df = pd.read_csv(DATA_PATH)
num_cols = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
df[num_cols] = scaler.transform(df[num_cols])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

y_true = test_df["Price"].values
y_pred = predictor.predict(test_df).values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# MAE: histogram błędów
errors = np.abs(y_true - y_pred)
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50, color="orange", edgecolor="black")
plt.title("Histogram of Absolute Errors (MAE)")
plt.xlabel("Absolute Error ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mae_hist.png")
plt.close()

# RMSE: wykres rozrzutu błędów kwadratowych
squared_errors = (y_true - y_pred) ** 2
plt.figure(figsize=(8, 5))
plt.scatter(y_true, squared_errors, alpha=0.5, color="skyblue")
plt.title("Squared Error vs True Price (RMSE)")
plt.xlabel("True Price ($)")
plt.ylabel("Squared Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/rmse_scatter.png")
plt.close()


# R2: predicted vs true
plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_pred, alpha=0.5, color="green", label="Predictions")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", label="Ideal")
plt.title("Predicted vs True Price (R²)")
plt.xlabel("True Price ($)")
plt.ylabel("Predicted Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/r2_fit.png")
plt.close()
