#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Case 3 Neural Network Script
Forward & Inverse Mapping + Plotting

Forward NN:  E, Y  -> hPA curve
Inverse NN: hPA curve -> E, Y
Both Anna (raw points) and Eric (averaged points) approaches implemented.
Plots are generated automatically in a separate folder.
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
n_bins_eric = 5  # number of bins for Eric averaging

# -----------------------------
# Output folders
# -----------------------------
NN_folder = "NN_Output"
forward_folder = os.path.join(NN_folder, "forward_predictions")
inverse_folder_anna = os.path.join(NN_folder, "inverse_anna")
inverse_folder_eric = os.path.join(NN_folder, "inverse_eric")

Plots_folder = "Plots"
forward_plot_folder = os.path.join(Plots_folder, "forward")
inverse_plot_anna_folder = os.path.join(Plots_folder, "inverse_anna")
inverse_plot_eric_folder = os.path.join(Plots_folder, "inverse_eric")

for folder in [NN_folder, forward_folder, inverse_folder_anna, inverse_folder_eric,
               Plots_folder, forward_plot_folder, inverse_plot_anna_folder, inverse_plot_eric_folder]:
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
X_params, Y_hPA = [], []                # Forward NN
X_hPA_anna, Y_params_anna = [], []      # Inverse NN Anna
X_hPA_eric, Y_params_eric = [], []      # Inverse NN Eric

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

    # Forward NN
    X_params.append([E, Y])
    Y_hPA.append(hPA_curve)

    # Inverse NN - Anna (raw points)
    X_hPA_anna.append(hPA_curve)
    Y_params_anna.append([E, Y])

    # Inverse NN - Eric (averaged)
    bin_size = len(hPA_curve) // n_bins_eric
    avg_curve = [sum(hPA_curve[i*bin_size:(i+1)*bin_size])/bin_size for i in range(n_bins_eric)]
    X_hPA_eric.append(avg_curve)
    Y_params_eric.append([E, Y])

# -----------------------------
# Scale data
# -----------------------------
# Forward NN scaling
param_scaler = StandardScaler().fit(X_params)
X_params_scaled = param_scaler.transform(X_params)

hPA_scaler = StandardScaler().fit(Y_hPA)
Y_hPA_scaled = hPA_scaler.transform(Y_hPA)

# Inverse NN scaling - Anna
param_scaler_inv = StandardScaler().fit(Y_params_anna)
X_hPA_anna_scaled = StandardScaler().fit(X_hPA_anna).transform(X_hPA_anna)
Y_params_anna_scaled = param_scaler_inv.transform(Y_params_anna)

# Inverse NN scaling - Eric
X_hPA_eric_scaled = StandardScaler().fit(X_hPA_eric).transform(X_hPA_eric)
Y_params_eric_scaled = param_scaler_inv.transform(Y_params_eric)

# -----------------------------
# Train Forward NN
# -----------------------------
forward_NN = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='lbfgs',
                          alpha=1e-5, max_iter=1000, random_state=42)
forward_NN.fit(X_params_scaled, Y_hPA_scaled)
print("Forward NN trained.")

# -----------------------------
# Train Inverse NN - Anna
# -----------------------------
inverse_NN_anna = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='lbfgs',
                               alpha=1e-5, max_iter=1000, random_state=42)
inverse_NN_anna.fit(X_hPA_anna_scaled, Y_params_anna_scaled)
print("Inverse NN (Anna) trained.")

# -----------------------------
# Train Inverse NN - Eric
# -----------------------------
inverse_NN_eric = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='lbfgs',
                               alpha=1e-5, max_iter=1000, random_state=42)
inverse_NN_eric.fit(X_hPA_eric_scaled, Y_params_eric_scaled)
print("Inverse NN (Eric) trained.")

# -----------------------------
# Save NN predictions and generate plots
# -----------------------------
for i, job_num in enumerate(job_numbers):

    # ---- Forward prediction ----
    hPA_pred_scaled = forward_NN.predict([X_params_scaled[i]])
    hPA_pred = hPA_scaler.inverse_transform(hPA_pred_scaled)[0]
    fwd_file = os.path.join(forward_folder, f"Job-{job_num:04d}_hPA_pred.dat")
    with open(fwd_file, "w") as f:
        f.write("Displacement    Force\n")
        for j in range(n_points_per_curve):
            f.write(f"{hPA_pred[2*j]:12.6e} {hPA_pred[2*j+1]:12.6e}\n")

    # Plot Forward NN curve
    plt.figure()
    plt.plot(hPA_pred[0::2], hPA_pred[1::2], 'b-', label='Predicted hPA')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title(f'Job-{job_num:04d} Forward NN')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(forward_plot_folder, f"Job-{job_num:04d}_hPA_pred.png"))
    plt.close()

    # ---- Inverse prediction Anna ----
    EY_pred_scaled = inverse_NN_anna.predict([X_hPA_anna_scaled[i]])
    EY_pred = param_scaler_inv.inverse_transform(EY_pred_scaled)[0]
    inv_file_anna = os.path.join(inverse_folder_anna, f"Job-{job_num:04d}_EY_pred_anna.dat")
    with open(inv_file_anna, "w") as f:
        f.write("YoungsModulus    YieldStress\n")
        f.write(f"{EY_pred[0]:12.6e} {EY_pred[1]:12.6e}\n")

    # Plot Inverse Anna prediction
    plt.figure()
    plt.bar(['E', 'Y'], EY_pred, color=['orange', 'green'])
    plt.ylabel('Predicted Value')
    plt.title(f'Job-{job_num:04d} Inverse NN (Anna)')
    plt.grid(axis='y')
    plt.savefig(os.path.join(inverse_plot_anna_folder, f"Job-{job_num:04d}_EY_pred_anna.png"))
    plt.close()

    # ---- Inverse prediction Eric ----
    EY_pred_scaled_eric = inverse_NN_eric.predict([X_hPA_eric_scaled[i]])
    EY_pred_eric = param_scaler_inv.inverse_transform(EY_pred_scaled_eric)[0]
    inv_file_eric = os.path.join(inverse_folder_eric, f"Job-{job_num:04d}_EY_pred_eric.dat")
    with open(inv_file_eric, "w") as f:
        f.write("YoungsModulus    YieldStress\n")
        f.write(f"{EY_pred_eric[0]:12.6e} {EY_pred_eric[1]:12.6e}\n")

    # Plot Inverse Eric prediction
    plt.figure()
    plt.bar(['E', 'Y'], EY_pred_eric, color=['orange', 'green'])
    plt.ylabel('Predicted Value')
    plt.title(f'Job-{job_num:04d} Inverse NN (Eric)')
    plt.grid(axis='y')
    plt.savefig(os.path.join(inverse_plot_eric_folder, f"Job-{job_num:04d}_EY_pred_eric.png"))
    plt.close()

print("NN predictions and plots saved for all jobs.")
print("Folders ready: NN_Output and Plots.")
