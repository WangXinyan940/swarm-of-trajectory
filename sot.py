import sys
import os
import json
import numpy as np
from jinja2 import Template

BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FS = 1e-15  # fs -> s
EH = 4.35974417e-18  # Hatree -> J
H = 6.626069934e-34
KB = 1.38064852e-23


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


def readMultiXYZ(text):
    """
    Read XYZ file with multi conformations.
    """
    pass


def readGauGrad(text):
    """
    Read Gaussian output and find energy gradient.
    """
    pass


def genMassMat(atom):
    """
    Generate matrix of mass.
    """
    massd = {"H": 1.008,
             "C": 12.011,
             "N": 14.007,
             "O": 15.999,
             "S": 32.066,
             "CL": 35.453,
             "BR": 79.904, }
    massv = np.array([massd[i.upper()] for i in atom])
    massm = np.zeros((len(atom), 3))
    massm[:, 0] = massv
    massm[:, 1] = massv
    massm[:, 2] = massv
    return massm


def genQMInput(atom, crd, temp, pre=False, nstep=-1):
    """
    Generate QM Input file for force calculation.
    """
    return temp.render(data=zip(atom, crd), pre=pre, nstep=nstep)


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
    print(genQMInput(t_atom, t_xyz, template, pre=True))


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


def calcGauGrad(atom, crd, template, nstep, path="g09"):
    """
    Calculate gradient using Gaussian.
    """
    with open("tmp.gjf", "w") as f:
        f.write(genQMInput(atom, crd, template,
                           pre=False if nstep == 0 else True, nstep=nstep))
    os.system("{} tmp.gjf".format(path))
    grad = readFile("tmp.log", readGauGrad)
    os.system("cp tmp.chk old.chk && rm tmp.gjf tmp.log tmp.chk")
    return grad


def setInitMotion(conf):
    """
    Set init coord and velocities. 
    """
    xyzs = readFile(conf["coordinate"], readMultiXYZ)
    if "velocity" in conf and conf["velocity"] is not None:
        vels = readFile(conf["velocity"], readMultiXYZ)
    else:
        vels = None
    if "start" not in conf or conf["start"] is None or conf["start"] > len(xyzs):
        atom, crd = xyzs[-1]
        crd = crd * ANGSTROM
        _, vel = vels[-1] if vels is not None else None, None
    else:
        rdm = np.random.randint(conf["start"], len(xyzs))
        crd = xyzs[rdm] * ANGSTROM
        _, vel = vels[rdm] if vels is not None else None, None
    if vel is None:
        T = conf["temperature"]
        massm = genMassMat(atom) * AMU
        vel = np.random.normal(0.0, np.sqrt(KB * T / massm))
    else:
        vel = vel * ANGSTROM / FS
    return atom, crd, vel


def main():
    """
    The main function.
    """
    printTitle()
    conf = argparse()
    # build template for qm engine
    template = readFile(conf["force"]["template"],
                        lambda x: Template("".join(x)))
    # select init crd and vel
    atom, crd, vel = setInitMotion(conf["init"])
    # run dynamics
    dynamics(atom, crd, vel, setting=conf["md"], out=conf[
             "print"], check=conf["check"], stop=conf["stop"])
    print("STDSP is finished.")


if __name__ == '__main__':
    main()
