import os
from odbAccess import *
from math import sqrt, pi

# -------------------------------
# Configuration
# -------------------------------

# Radius of spherical indenter
R = 5.0

# Folder where parent script stored all jobs
JOB_FOLDER = "Jobs_Batch"

# Folder for rpt files
rpt_folder = "RPT_Files"
if not os.path.exists(rpt_folder):
    os.makedirs(rpt_folder)

# -------------------------------
# Function to read data from a single job
# -------------------------------
def read_hPA(job):
    # Open rpt file
    rpt = open(os.path.join(rpt_folder, job + ".rpt"), "w")
    rpt.write("Time         Displacement        Force  ContactArea \n")

    # Open odb from batch folder
    odb = openOdb(os.path.join(JOB_FOLDER, job + ".odb"))

    # Reference node for indenter
    refNSet = odb.rootAssembly.instances['PART-1-1'].nodeSets['SPHERE']

    # Field output frames
    frames = odb.steps['Step-1'].frames
    Nframes = len(frames)

    # History output
    HREG = odb.steps['Step-1'].historyRegions['NodeSet  Z000001']
    CAREA = HREG.historyOutputs['CAREA    SURF/RSURF'].data

    t_avg = 0.0
    t_max = 0.0

    for i in range(Nframes):
        ftime        = frames[i].frameValue
        displacement = -frames[i].fieldOutputs['U'].getSubset(region=refNSet).values[0].data[1]
        force        = -frames[i].fieldOutputs['RF'].getSubset(region=refNSet).values[0].data[1]
        Ctime        = CAREA[i][0]
        contactArea  = CAREA[i][1]

        t_avg += ftime
        if ftime > t_max:
            t_max = ftime

        if ftime != Ctime:
            print("Warning: Increment %d - Time mismatch" % i)
            print("History time:", Ctime, "Field time:", ftime)

        rpt.write("%12.6e %12.6e %12.6e %12.6e\n"
                  % (ftime, displacement, force, contactArea))

    t_avg /= float(Nframes)

    odb.close()
    rpt.close()

    return t_avg, t_max


# -------------------------------
# Main script: automatically detect jobs
# -------------------------------

# Detect all Job-XXXX.odb files in batch folder
job_files = sorted([
    f for f in os.listdir(JOB_FOLDER)
    if f.startswith("Job-") and f.endswith(".odb")
])

# Open database file
Database = open("Database-R.dat", "w")
Database.write("JobName      AvgTime       MaxTime\n")

# Loop through all detected jobs
for job_file in job_files:
    job_name = job_file.replace(".odb", "")
    print("Reading job:", job_name)

    t_avg, t_max = read_hPA(job_name)
    Database.write("%s %12.6e %12.6e\n" % (job_name, t_avg, t_max))

Database.close()

print("All jobs processed.")
print("RPT files saved in folder:", rpt_folder)
