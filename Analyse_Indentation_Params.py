# Analyse_Indentation_Params.py (with procedural array generation)
from ABAQUSJob import *
import os  # needed to call cousin script

# -----------------------------
# Global input variables
# -----------------------------
num_E = 5          # number of Young's modulus values
num_Y = 5          # number of Yield stress values

E_start, E_end = 40000.0, 60000.0  # Young's modulus range
Y_start, Y_end = 80.0, 120.0       # Yield stress range

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
#Database.write("JobNumber      YoungsModulus       YieldStress\n")

# Initialize job counter
JobNumber = 0

# -----------------------------
# Nested loop to run multiple jobs
# -----------------------------
for E in YoungsModulus_list:       # Loop over Young's modulus
    for Y in YieldStress_list:     # Loop over yield stress
        JobNumber += 1

        # Copy template input file into txt object
        JobInp = "TemplateJob.inp"
        with open(JobInp, "r") as inp:
            txt = inp.read()

        # Replace reserved space for material parameters with chosen values
        E_txt = "%12.1f" % E
        Y_txt = "%12.1f" % Y
        txt = txt.replace("YOUNGSMODULUS", E_txt)
        txt = txt.replace("YIELDSTRESS", Y_txt)

        # Create jobname for submission and write info to screen
        Filename = "Job-%02d.inp" % JobNumber
        print("Filename:", Filename)
        print("Job %d:, E = %f, sigma_y = %f" % (JobNumber, E, Y))

        # Write input file for job submission
        with open(Filename, "w") as out:
            out.write(txt)

        # Save values in database
        Database.write("%2d %12.6e %12.6e\n" % (JobNumber, E, Y))

        # Submit Abaqus job
        prefix  = "Job-%02d" % JobNumber
        submitAbaqusJob(prefix)
        print("Job %d completed.\n" % JobNumber)

# Close database file
Database.close()

# -----------------------------
# Automatically call cousin script to process outputs
# -----------------------------
print("All Abaqus jobs finished. Running Read_hPA.py to generate .rpt files...\n")
os.system("abaqus python Read_hPA.py")
print("Cousin script completed. Workflow fully automated!")

