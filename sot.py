import numpy as np
import sys
import json


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
        python sot.py -j json
    """)


def argparse():
    """
    Parse the args.
    """
    if "-j" not in sys.argv:
        printHelp()
        exit()
    with open(sys.argv[2], "r") as f:
        text = "".join(f)
    return json.loads(text)


def main():
    """
    The main function.
    """
    printTitle()
    conf = argparse()


if __name__ == '__main__':
    main()
