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
RPT_folder = "RPT_Files"       # Folder containing Job-XX.rpt
Database_file = "Database-P.dat"  # Anchoring E & Y values
n_points_per_curve = 50          # Number of hPA points per job (assume uniform)
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
X_hPA = []   # hPA curves -> inverse NN input
Y_params = [] # E,Y -> inverse NN target, forward NN input
X_params = [] # E,Y -> forward NN input
Y_hPA = []   # hPA curves -> forward NN target

for job_num, E, Y in zip(job_numbers, E_values, Y_values):
    rpt_file = os.path.join(RPT_folder, f"Job-{job_num:02d}.rpt")
    if not os.path.exists(rpt_file):
        print(f"Warning: {rpt_file} not found. Skipping job {job_num}.")
        continue
    
    # Read hPA points from rpt
    hPA_curve = []
    with open(rpt_file, "r") as f:
        next(f)  # skip header line
        for i, line in enumerate(f):
            if i >= n_points_per_curve:
                break
            parts = line.split()
            displacement = float(parts[1])
            force = float(parts[2])
            # Optional: concatenate displacement & force into one feature
            hPA_curve.append(displacement)
            hPA_curve.append(force)
    
    if len(hPA_curve) != 2*n_points_per_curve:
        print(f"Warning: Job-{job_num:02d}.rpt has {len(hPA_curve)} points instead of expected {2*n_points_per_curve}.")
        continue

    # Forward NN
    X_params.append([E, Y])
    Y_hPA.append(hPA_curve)

    # Inverse NN
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
# Train Forward NN (E,Y -> hPA)
# -----------------------------
forward_NN = MLPRegressor(
    hidden_layer_sizes=(10,),   # small network
    activation='relu',
    solver='lbfgs',
    alpha=1e-5,
    max_iter=1000,
    random_state=42
)
forward_NN.fit(X_params_scaled, Y_hPA_scaled)
print("Forward NN trained.")

# -----------------------------
# Train Inverse NN (hPA -> E,Y)
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
# Example prediction
# -----------------------------
# Forward example
#E_test, Y_test = 45000.0, 95.0
#X_test_scaled = param_scaler.transform([[E_test, Y_test]])
#hPA_pred_scaled = forward_NN.predict(X_test_scaled)
#hPA_pred = hPA_scaler.inverse_transform(hPA_pred_scaled)
#print(f"Forward prediction for E={E_test}, Y={Y_test} done.")

# Inverse example
#hPA_example = hPA_pred[0]
#X_hPA_example_scaled = hPA_scaler.transform([hPA_example])
#E_Y_pred_scaled = inverse_NN.predict(X_hPA_example_scaled)
#E_Y_pred = param_scaler.inverse_transform(E_Y_pred_scaled)
#print(f"Inverse prediction from forward output: E={E_Y_pred[0,0]:.2f}, Y={E_Y_pred[0,1]:.2f}")

#print("Done.")

