# ============================================================
# Baseline ML Model for Degree of Cure Prediction
# Model: MLP Regressor
#
# What it does:
# Learns the relationship between Temperature and Degree of Cure (DoC)
# using Temperature as the only input feature.
#
# Input:  Temperature
# Output: Predicted Degree of Cure (DoC)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------------------------------------------
# 1. Load data from one folder
# ------------------------------------------------------------

folder = "/Users/anastasia/Desktop/artificial_sensor_dataset"   # or the other one

csv_files = sorted(glob.glob(os.path.join(folder, "sample_*.csv")))

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)
    temp_df["sample_id"] = os.path.splitext(os.path.basename(file))[0]
    temp_df["source_file"] = os.path.basename(file)
    df_list.append(temp_df)

if not df_list:
    raise ValueError(f"No CSV files found in folder: {folder}")

# Combine everything
df = pd.concat(df_list, ignore_index=True)

# Clean index just in case
df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(csv_files)} files from '{folder}'")
print("Shape:", df.shape)
print("First few rows:")
print(df.head())
print()
print("Columns:")
print(df.columns.tolist())

required_cols = ["Temperature", "Time", "Degree_of_Cure"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ------------------------------------------------------------
# 2. Select feature and target
# ------------------------------------------------------------
# Expected columns:
#   Temperature
#   DoC
#   Time

X = df[["Temperature", "Degree_of_Cure", "dT_dt"]].values
y = df["Degree_of_Cure"].values

# ------------------------------------------------------------
# 3. Train/test split (BY SAMPLE, NOT ROW)
# ------------------------------------------------------------
sample_ids = df["sample_id"].unique()

train_ids, test_ids = train_test_split(
    sample_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df["sample_id"].isin(train_ids)].copy()
test_df  = df[df["sample_id"].isin(test_ids)].copy()

X_train = train_df[["Temperature", "Time"]].values
y_train = train_df["Degree_of_Cure"].values

X_test = test_df[["Temperature", "Time"]].values
y_test = test_df["Degree_of_Cure"].values

# ------------------------------------------------------------
# 4. Scale input data
# ------------------------------------------------------------
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# ------------------------------------------------------------
# 5. Create the model
# ------------------------------------------------------------
model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation='relu',
    solver='adam',
    alpha=1e-2,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

# ------------------------------------------------------------
# 6. Train the model
# ------------------------------------------------------------
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 7. Predict on test set
# ------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
train_pred = model.predict(X_train_scaled)

# ------------------------------------------------------------
# 8. Evaluate
# ------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model: MLP Regressor")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R^2  = {r2:.6f}")

print("\nOverfitting check:")
print(f"Train R^2 = {r2_score(y_train, train_pred):.6f}")
print(f"Test  R^2 = {r2:.6f}")

# ------------------------------------------------------------
# 9. Plot actual vs predicted
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    linewidth=1
)
plt.xlabel("Actual DoC")
plt.ylabel("Predicted DoC")
plt.title("Predicted vs Actual DoC Across Test Data (sorted by Temperature)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Plot DoC vs Temperature
# ------------------------------------------------------------
test_results = pd.DataFrame({
    "Temperature": X_test[:, 0],
    "Actual_DoC": y_test,
    "Predicted_DoC": y_pred
}).sort_values("Temperature")

plt.figure(figsize=(8, 5))
plt.plot(
    test_results["Temperature"],
    test_results["Actual_DoC"],
    'o-',
    label="Actual DoC"
)
plt.plot(
    test_results["Temperature"],
    test_results["Predicted_DoC"],
    's-',
    label="Predicted DoC"
)
plt.xlabel("Temperature")
plt.ylabel("Degree of Cure")
plt.title("MLP Regressor: DoC vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. Predict for new temperature-time pairs
# ------------------------------------------------------------
new_inputs = np.array([
    [40, 10],
    [60, 20],
    [80, 30],
    [100, 40],
    [120, 50]
])

new_inputs_scaled = x_scaler.transform(new_inputs)
new_predictions = model.predict(new_inputs_scaled)

print("\nPredictions for new temperature-time pairs:")
for row, pred in zip(new_inputs, new_predictions):
    T, t = row
    print(f"Temperature = {T:6.2f}, Time = {t:6.2f} -> Predicted DoC = {pred:.6f}")


