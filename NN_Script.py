#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Case 3 Neural Network Script
Forward & Inverse Mapping

Forward NN:  E, Y  -> hPA curve
Inverse NN: hPA curve -> E, Y
"""

# -----------------------------
# Import libraries
# -----------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

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
forward_folder = os.path.join(NN_folder, "forward_predictions")
inverse_folder = os.path.join(NN_folder, "inverse_predictions")

for folder in [NN_folder, forward_folder, inverse_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# -----------------------------
# Read Database-P.dat
# -----------------------------
job_numbers = []
E_values = []
Y_values = []

with open(Database_file, "r") as f:
    for line in f:
        parts = line.split()
        job_numbers.append(int(parts[0]))
        E_values.append(float(parts[1]))
        Y_values.append(float(parts[2]))

# -----------------------------
# Read RPT files & build dataset
# -----------------------------
X_hPA = []
Y_params = []
X_params = []
Y_hPA = []

for job_num, E, Y in zip(job_numbers, E_values, Y_values):
    rpt_file = os.path.join(RPT_folder, f"Job-{job_num:04d}.rpt")
    if not os.path.exists(rpt_file):
        print(f"Warning: {rpt_file} not found. Skipping.")
        continue

    hPA_curve = []
    with open(rpt_file, "r") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n_points_per_curve:
                break
            parts = line.split()
            displacement = float(parts[1])
            force = float(parts[2])
            hPA_curve.extend([displacement, force])

    if len(hPA_curve) != 2 * n_points_per_curve:
        print(f"Warning: Job-{job_num:04d} has incomplete data.")
        continue

    X_params.append([E, Y])
    Y_hPA.append(hPA_curve)
    X_hPA.append(hPA_curve)
    Y_params.append([E, Y])

# -----------------------------
# Scale data
# -----------------------------
param_scaler = StandardScaler().fit(X_params)
X_params_scaled = param_scaler.transform(X_params)
Y_params_scaled = param_scaler.transform(Y_params)

hPA_scaler = StandardScaler().fit(Y_hPA)
Y_hPA_scaled = hPA_scaler.transform(Y_hPA)
X_hPA_scaled = hPA_scaler.transform(X_hPA)

# -----------------------------
# Train Forward NN
# -----------------------------
forward_NN = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='lbfgs',
    alpha=1e-5,
    max_iter=1000,
    random_state=42
)
forward_NN.fit(X_params_scaled, Y_hPA_scaled)
print("Forward NN trained.")

# -----------------------------
# Train Inverse NN
# -----------------------------
inverse_NN = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='lbfgs',
    alpha=1e-5,
    max_iter=1000,
    random_state=42
)
inverse_NN.fit(X_hPA_scaled, Y_params_scaled)
print("Inverse NN trained.")

# -----------------------------
# Save NN predictions for ALL jobs
# -----------------------------
for i, job_num in enumerate(job_numbers):

    # ---- Forward prediction ----
    hPA_pred_scaled = forward_NN.predict([X_params_scaled[i]])
    hPA_pred = hPA_scaler.inverse_transform(hPA_pred_scaled)[0]

    forward_file = os.path.join(forward_folder, f"Job-{job_num:04d}_hPA_pred.dat")
    with open(forward_file, "w") as f:
        f.write("Displacement    Force\n")
        for j in range(n_points_per_curve):
            f.write(f"{hPA_pred[2*j]:12.6e} {hPA_pred[2*j+1]:12.6e}\n")

    # ---- Inverse prediction ----
    EY_pred_scaled = inverse_NN.predict([X_hPA_scaled[i]])
    EY_pred = param_scaler.inverse_transform(EY_pred_scaled)[0]

    inverse_file = os.path.join(inverse_folder, f"Job-{job_num:04d}_EY_pred.dat")
    with open(inverse_file, "w") as f:
        f.write("YoungsModulus    YieldStress\n")
        f.write(f"{EY_pred[0]:12.6e} {EY_pred[1]:12.6e}\n")

print("NN predictions saved successfully.")
print("NN_Output folder ready for plotting.")
