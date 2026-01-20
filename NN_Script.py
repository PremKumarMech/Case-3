#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Case 3 Neural Network Script - Forward NN Only

- Forward NN: E, Y -> hPA curve
- Y is fixed (e.g., 100 MPa)
- All forward curves are saved in NN_Output
- Single plot visualizes all predicted curves with color representing E
"""

# -----------------------------
# Import libraries
# -----------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# -----------------------------
# Global parameters
# -----------------------------
RPT_folder = "RPT_Files"
Database_file = "Database-P.dat"
n_points_per_curve = 50

# -----------------------------
# Output folders
# -----------------------------
NN_folder = "NN_Output"
plots_folder = "Plots"

for folder in [NN_folder, plots_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# -----------------------------
# Read Database-P.dat
# -----------------------------
job_numbers, E_values, Y_values = [], [], []

with open(Database_file, "r") as f:
    for line in f:
        parts = line.split()
        job_numbers.append(int(parts[0]))
        E_values.append(float(parts[1]))
        Y_values.append(float(parts[2]))

# -----------------------------
# Read RPT files & build dataset
# -----------------------------
X_params, Y_hPA = [], []

for job_num, E, Y in zip(job_numbers, E_values, Y_values):
    rpt_file = os.path.join(RPT_folder, f"Job-{job_num:04d}.rpt")
    if not os.path.exists(rpt_file):
        print(f"Warning: {rpt_file} not found. Skipping job {job_num}.")
        continue

    hPA_curve = []
    with open(rpt_file, "r") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i >= n_points_per_curve:
                break
            parts = line.split()
            displacement = float(parts[1])
            force = float(parts[2])
            hPA_curve.extend([displacement, force])

    if len(hPA_curve) != 2 * n_points_per_curve:
        print(f"Warning: Job-{job_num:04d} has incomplete data. Skipping.")
        continue

    # Forward NN dataset
    X_params.append([E, Y])
    Y_hPA.append(hPA_curve)

# -----------------------------
# Scale data
# -----------------------------
param_scaler = StandardScaler().fit(X_params)
X_params_scaled = param_scaler.transform(X_params)

hPA_scaler = StandardScaler().fit(Y_hPA)
Y_hPA_scaled = hPA_scaler.transform(Y_hPA)

# -----------------------------
# Train Forward NN
# -----------------------------
forward_NN = MLPRegressor(hidden_layer_sizes=(10,), activation='relu',
                          solver='lbfgs', alpha=1e-5, max_iter=1000,
                          random_state=42)
forward_NN.fit(X_params_scaled, Y_hPA_scaled)
print("Forward NN trained.")

# -----------------------------
# Save NN predictions and prepare plot
# -----------------------------
plt.figure(figsize=(8,6))
colors = plt.cm.viridis([i/len(X_params) for i in range(len(X_params))])  # color gradient

for i, job_num in enumerate(job_numbers):
    hPA_pred_scaled = forward_NN.predict([X_params_scaled[i]])
    hPA_pred = hPA_scaler.inverse_transform(hPA_pred_scaled)[0]

    # Save NN .dat file
    fwd_file = os.path.join(NN_folder, f"Job-{job_num:04d}_hPA_pred.dat")
    with open(fwd_file, "w") as f:
        f.write("Displacement    Force\n")
        for j in range(n_points_per_curve):
            f.write(f"{hPA_pred[2*j]:12.6e} {hPA_pred[2*j+1]:12.6e}\n")

    # Plot forward curve
    plt.plot(hPA_pred[0::2], hPA_pred[1::2], color=colors[i], label=f"E={E_values[i]:.0f}")

plt.xlabel("Displacement")
plt.ylabel("Force")
plt.title("Forward NN Predictions - All Jobs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "Forward_curves_all_jobs.png"))
plt.close()

print("All NN .dat files saved in NN_Output.")
print("Forward curves plot saved in Plots/Forward_curves_all_jobs.png")
