from os import access, R_OK, path, system, remove
from time import sleep

""" 
# Python library for Abaqus job submission including functions:
    submitAbaqusJob( jobName )
        runAbaqus(jobName)
        waitForStaFile( staFileName )
        waitForOdbAccess( lckFileName )
        checkSuccessfulCompletion( staFileName )
            tailStatusFile( staFileName )
        cleanUp( jobName )
#
"""

def submitAbaqusJob( jobName ):
    runAbaqus( jobName )
    staFileName = jobName + ".sta"
    lckFileName = jobName + ".lck"

    waitForStaFile( staFileName )
    waitForOdbAccess( lckFileName )

    successfully_ended = checkSuccessfulCompletion( staFileName )
    if successfully_ended:
        cleanUp( jobName )

def runAbaqus(jobName):
    # run abaqus simulation
    abaqus_command = "abaqus job=" + jobName
    print("submitting job:", abaqus_command)
    system( abaqus_command )

def waitForStaFile( staFileName ):
    # wait until .sta file exists
    print("waiting for status file.",)
    while not access( staFileName, R_OK ):
        sleep( 10 )
    print()

def waitForOdbAccess( lckFileName ):
    # wait until .odb file has been closed successfully,
    print("Waiting for odb access ",)
    while path.exists( lckFileName ):
        sleep( 1 )

def checkSuccessfulCompletion( staFileName ):
    sta_file = open( staFileName, "r" )
    sta_txt = sta_file.read()
    sta_file.close()

    if sta_txt.find( "THE ANALYSIS HAS NOT BEEN COMPLETED" ) > -1:
        successfully_ended = False
        print("Analysis has not been completed.")
        return successfully_ended 
    else:
        successfully_ended = True
        print("Analysis has been completed.")
        return successfully_ended 

def tailStatusFile( staFileName ):
    sta_file = open( staFileName, "r" )
    sta_txt = sta_file.read()
    sta_file.close()
    allLines = sta_txt.split('\n')
    lastLine = allLines[-1]
    return lastLine,

def cleanUp( jobName ):
    suffixList = [ "log", "prt", "msg", "sta", "com", "dat" ]
    for s in suffixList:
        remove (jobName+"."+s) 

