import numpy as np
import sys


def printTitle():
    """
    Print the title of the output result.
    """
    print("""
    --------------------------------------------------
    | Swarm-of-Trajectory Dynamic Simulation Package
    | 
    | Version: 0.1
    | Program Language: Python 3.6
    | Developer: Xinyan Wang
    --------------------------------------------------
    """)


def printHelp():
    """
    Print help.
    """
    print("""
    Usage:
        python sot.py 

    -n --name      job name. Use name of input coord file by default.
    --xin          input coordinate
    --vin          input velocities
    --xout         output coordinate
    --vout         output velocities
    --nout         frequency of output details
    --nvt          constraint nvt
    -t --temp      initial temperature
    --nve          nve dynamics
    --nsteps       num of steps
    --dt           time step (fs)
    -h --help      help
    """)


def argparse():
    """
    Parse the args.
    """
    if "-h" in sys.argv or "--help" in sys.argv:
        printHelp()
        exit()
    for i in sys.argv[1:]:
        print(i)


def main():
    """
    The main function.
    """
    printTitle()
    conf = argparse()


if __name__ == '__main__':
    main()
