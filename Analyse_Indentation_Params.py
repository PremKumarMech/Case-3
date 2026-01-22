# Analyse_Indentation_Params.py (with procedural array generation)
from ABAQUSJob import *
import os
import shutil   # needed for folderisation

# -----------------------------
# Global input variables
# -----------------------------
num_E = 5          # number of Young's modulus values
num_Y = 5          # number of Yield stress values

E_start, E_end = 190000.0, 210000.0  # Young's modulus range
Y_start, Y_end = 250.0, 500.0       # Yield stress range

# -----------------------------
# Batch folder for all jobs
# -----------------------------
BATCH_FOLDER = "Jobs_Batch"

if not os.path.exists(BATCH_FOLDER):
    os.mkdir(BATCH_FOLDER)

# -----------------------------
# Generate arrays without NumPy
# -----------------------------
def linspace(start, end, num):
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i*step for i in range(num)]

YoungsModulus_list = linspace(E_start, E_end, num_E)
YieldStress_list   = linspace(Y_start, Y_end, num_Y)

# -----------------------------
# Open database file
# -----------------------------
Database = open("Database-P.dat", "w")

# Initialize job counter
JobNumber = 0

# -----------------------------
# Nested loop to run multiple jobs
# -----------------------------
for E in YoungsModulus_list:
    for Y in YieldStress_list:
        JobNumber += 1

        # Copy template input file into txt object
        JobInp = "TemplateJob.inp"
        with open(JobInp, "r") as inp:
            txt = inp.read()

        # Replace reserved space for material parameters
        E_txt = "%12.1f" % E
        Y_txt = "%12.1f" % Y
        txt = txt.replace("YOUNGSMODULUS", E_txt)
        txt = txt.replace("YIELDSTRESS", Y_txt)

        # ---- UPDATED JOB NAMING (4 digits) ----
        Filename = "Job-%04d.inp" % JobNumber
        prefix   = "Job-%04d" % JobNumber

        print("Filename:", Filename)
        print("Job %d:, E = %f, sigma_y = %f" % (JobNumber, E, Y))

        # Write input file
        with open(Filename, "w") as out:
            out.write(txt)

        # Save values in database
        Database.write("%2d %12.6e %12.6e\n" % (JobNumber, E, Y))

        # Submit Abaqus job
        submitAbaqusJob(prefix)
        print("Job %d completed.\n" % JobNumber)

        # ---- MOVE FILES TO BATCH FOLDER ----
        inp_file = prefix + ".inp"
        odb_file = prefix + ".odb"

        if os.path.exists(inp_file):
            shutil.move(inp_file, os.path.join(BATCH_FOLDER, inp_file))

        if os.path.exists(odb_file):
            shutil.move(odb_file, os.path.join(BATCH_FOLDER, odb_file))

# Close database file
Database.close()

# -----------------------------
# Automatically call cousin script
# -----------------------------
print("All Abaqus jobs finished. Running Read_hPA.py to generate .rpt files...\n")
os.system("abaqus python Read_hPA.py")
print("Cousin script completed. Workflow fully automated!")

# -----------------------------
# 3rd Child: Neural Network Script
# -----------------------------
NN_script = "NN_Script.py"  # make sure this script exists in the same folder

if os.path.exists(NN_script):
    print("Launching 3rd child: Neural Network Script...")
    os.system(f"python3 {NN_script}")
    print("3rd child finished. All workflows completed!")
else:
    print(f"Error: {NN_script} not found. Make sure NN_Script.py is in the folder.")

