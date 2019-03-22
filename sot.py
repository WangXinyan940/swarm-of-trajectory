import sys
import json
import numpy as np
from jinja2 import Template


def printTitle():
    """
    Print the title of the output result.
    """
    print("""
----------------------------------------------------------------
| Swarm-of-Trajectory Dynamic Simulation Package
| 
| Version: 0.1
| Program Language: Python 3.6
| Developer: Xinyan Wang
| Homepage: https://github.com/WangXinyan940/swarm-of-trajectory
----------------------------------------------------------------
    """)


def printHelp():
    """
    Print help.
    """
    print("""
    Usage:
        python sot.py -j json -t template
    """)


def readFile(fname, func):
    """
    Read text file and deal with different functions.
    """
    with open(fname, "r") as f:
        text = f.readlines()
    return func(text)


def readXYZ(text):
    """
    Read xyz format text.
    """
    natoms = int(text[0].strip())
    body = text[2:natoms + 2]
    body = [i.strip().split() for i in body]
    atom = [i[0] for i in body]
    crd = [[float(j) for j in i[1:]] for i in body]
    return atom, crd

def genQMInput(atom, crd, temp):
    """
    Generate QM Input file for force calculation.
    """
    return temp.render(data=zip(atom, crd))


def testTemplate(conf):
    """
    Test whether the template file is correct. (Use water)
    """
    t_atom = ["H", "O", "H"]
    t_xyz = [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
    template = readFile(conf["force"]["template"],
                        lambda x: Template("".join(x)))
    print("Generate template below:\n++++++++++++++++++++++++")
    print(genQMInput(t_atom, t_xyz, template))


def argparse():
    """
    Parse the args.
    """
    # Test qm template
    if "-j" not in sys.argv and "-t" not in sys.argv:
        printHelp()
        exit()

    with open(sys.argv[2], "r") as f:
        text = "".join(f)
        conf = json.loads(text)

    if "-t" in sys.argv:
        testTemplate(conf)
        exit()
    return conf


def main():
    """
    The main function.
    """
    printTitle()
    conf = argparse()


if __name__ == '__main__':
    main()
