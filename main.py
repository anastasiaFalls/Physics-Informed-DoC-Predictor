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
# 1. Load data from MULTIPLE folders
# ------------------------------------------------------------
folders = [
    "/Users/anastasia/Desktop/artificial_sensor_dataset",
    "/Users/anastasia/Desktop/artificial_sensor_dataset_with_inlet"
]

df_list = []
total_files = 0

for folder in folders:
    csv_files = sorted(glob.glob(os.path.join(folder, "sample_*.csv")))

    print(f"{folder}: found {len(csv_files)} files")
    total_files += len(csv_files)

    for file in csv_files:
        temp_df = pd.read_csv(file)

        # Make sample_id unique across folders
        temp_df["sample_id"] = (
            os.path.basename(folder) + "_" +
            os.path.splitext(os.path.basename(file))[0]
        )

        temp_df["source_file"] = os.path.basename(file)
        temp_df["source_folder"] = os.path.basename(folder)

        df_list.append(temp_df)

# Safety check
if not df_list:
    raise ValueError("No CSV files found in any folder.")

# Combine everything
df = pd.concat(df_list, ignore_index=True)
df.reset_index(drop=True, inplace=True)

print(f"\nLoaded {total_files} files total")
print("Shape:", df.shape)
print("First few rows:")
print(df.head())
print()
print("Columns:")
print(df.columns.tolist())
print("Unique samples:", df["sample_id"].nunique())

# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------
required_cols = ["Temperature", "Degree_of_Cure", "Sensor_Value", "sample_id", "Time"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ------------------------------------------------------------
# 2. Create next-step target
# ------------------------------------------------------------
df = df.sort_values(["sample_id", "Time"]).copy()

df["DoC_next"] = df.groupby("sample_id")["Degree_of_Cure"].shift(-1)

# Drop last row of each sample, since DoC_next is undefined there
df = df.dropna(subset=["DoC_next"]).copy()

print("\nAfter creating DoC_next:")
print(df[["sample_id", "Time", "Temperature", "Sensor_Value", "Degree_of_Cure", "DoC_next"]].head())

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

train_df = train_df.sample(n=200000, random_state=42)

print("\nTraining samples:", len(train_ids))
print("Testing samples:", len(test_ids))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# ------------------------------------------------------------
# 4. Select features and target
#    Model: DoC_next = f(Temperature, Degree_of_Cure, Sensor_Value)
# ------------------------------------------------------------
feature_cols = ["Temperature", "Degree_of_Cure", "Sensor_Value", "Time"]
target_col = "DoC_next"

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ------------------------------------------------------------
# 5. Scale input data
# ------------------------------------------------------------
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# ------------------------------------------------------------
# 6. Create the model
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
# 7. Train the model
# ------------------------------------------------------------
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 8. Predict
# ------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
train_pred = model.predict(X_train_scaled)

# ------------------------------------------------------------
# 9. Evaluate
# ------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel: MLP Regressor")
print("Task: Predict next-step Degree of Cure")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R^2  = {r2:.6f}")

print("\nOverfitting check:")
print(f"Train R^2 = {r2_score(y_train, train_pred):.6f}")
print(f"Test  R^2 = {r2:.6f}")

# ------------------------------------------------------------
# 10. Naive baseline: DoC_next = current DoC
# ------------------------------------------------------------
baseline_pred = X_test[:, 1]  # current Degree_of_Cure

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nNaive baseline: DoC_next = current DoC")
print(f"Baseline MAE  = {baseline_mae:.6f}")
print(f"Baseline RMSE = {baseline_rmse:.6f}")
print(f"Baseline R^2  = {baseline_r2:.6f}")

# ------------------------------------------------------------
# 11. Plot actual vs predicted next-step DoC
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    linewidth=1
)
plt.xlabel("Actual DoC_next")
plt.ylabel("Predicted DoC_next")
plt.title("MLP Regressor: Actual vs Predicted Next-Step Degree of Cure")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. Plot predicted next-step DoC vs current DoC
# ------------------------------------------------------------
test_results = pd.DataFrame({
    "Current_DoC": X_test[:, 1],
    "Actual_DoC_next": y_test,
    "Predicted_DoC_next": y_pred
}).sort_values("Current_DoC")

plt.figure(figsize=(8, 5))
plt.plot(
    test_results["Current_DoC"],
    test_results["Actual_DoC_next"],
    'o',
    alpha=0.3,
    label="Actual DoC_next"
)
plt.plot(
    test_results["Current_DoC"],
    test_results["Predicted_DoC_next"],
    'o',
    alpha=0.3,
    label="Predicted DoC_next"
)
plt.xlabel("Current Degree of Cure")
plt.ylabel("Next-Step Degree of Cure")
plt.title("Next-Step Cure Prediction vs Current Cure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. Plot S-curves using RECURSIVE rollout
#     (model feeds on its own predicted DoC)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

# pick a few random test samples
sample_ids_to_plot = np.random.choice(
    test_df["sample_id"].unique(),
    size=5,
    replace=False
)

for sid in sample_ids_to_plot:
    sample = test_df[test_df["sample_id"] == sid].copy()
    sample = sample.sort_values("Time").reset_index(drop=True)

    time_vals = sample["Time"].values
    temp_vals = sample["Temperature"].values
    sensor_vals = sample["Sensor_Value"].values
    actual_doc = sample["Degree_of_Cure"].values

    # Start rollout from the TRUE initial DoC
    current_doc = actual_doc[0]

    predicted_doc_curve = [current_doc]

    # Step forward recursively
    for i in range(1, len(sample)):
        x_input = np.array([[temp_vals[i-1], current_doc, sensor_vals[i-1], time_vals[i-1]]])
        x_input_scaled = x_scaler.transform(x_input)

        next_doc_pred = model.predict(x_input_scaled)[0]

         # --- PHYSICS CONSTRAINTS ---
        next_doc_pred = max(next_doc_pred, current_doc)  # no decrease
        next_doc_pred = min(next_doc_pred, 100)          # cap at full cure

        # Optional safety clamp to keep DoC physical
        next_doc_pred = np.clip(next_doc_pred, 0, 100)

        predicted_doc_curve.append(next_doc_pred)

        # Feed prediction back in
        current_doc = next_doc_pred

    predicted_doc_curve = np.array(predicted_doc_curve)

    # Plot actual
    actual_line, = plt.plot(
        time_vals,
        actual_doc,
        linewidth=2,
        label=f"{sid} Actual"
    )

    # Plot recursive prediction in matching color
    plt.plot(
        time_vals,
        predicted_doc_curve,
        linestyle='--',
        linewidth=2,
        color=actual_line.get_color(),
        label=f"{sid} Predicted"
    )

plt.xlabel("Time")
plt.ylabel("Degree of Cure")
plt.title("Actual vs Predicted Cure Evolution (Recursive Rollout)")
plt.grid(True)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

