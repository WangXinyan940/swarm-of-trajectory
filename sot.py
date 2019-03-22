import sys
import os
import json
import numpy as np
from jinja2 import Template
NAME = ""
BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FS = 1e-15  # fs -> s
EH = 4.35974417e-18  # Hartrees -> J
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


def readGauGrad(text, natoms):
    """
    Read Gaussian output and find energy gradient.
    """
    ener = [i for i in text if "SCF Done:" in i]
    if len(ener) != 0:
        ener = ener[-1]
        ener = np.float64(ener.split()[4])
    else:
        ener = np.float64([i for i in text if "Energy=" in i][-1].split()[1])
    for ni, li in enumerate(text):
        if "Forces (Hartrees/Bohr)" in li:
            break
    forces = text[ni + 3:ni + 3 + natoms]
    forces = [i.strip().split()[-3:] for i in forces]
    forces = [[np.float64(i[0]), np.float64(i[1]), np.float64(i[2])]
              for i in forces]
    return ener * EH, -np.array(forces) * EH / BOHR


def writeXYZ():
    pass


def distance(crd, i, j):
    """
    Calc distance of two points.
    """
    return np.sqrt(((crd[i,:] - crd[j,:]) ** 2).sum())


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
    print(">>> Generate template below:\n++++++++++++++++++++++++++++")
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
    grad = readFile("tmp.log", lambda x: readGauGrad(x, len(atom)))
    os.system("cp tmp.chk old.chk && rm tmp.gjf tmp.log tmp.chk")
    return grad


def genGrad(conf, template):
    if conf["engine"].upper() == "GAUSSIAN":
        return lambda atom, crd, nstep: calcGauGrad(atom, crd / ANGSTROM, template, nstep, conf["path"])


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


def dynamics(atom, initx, initv, grad=None, conf=None):
    """
    Run dynamics. Using Velocity verlet algorithm.
    """
    md, prt, chk, stop = conf["md"], conf["print"], conf["check"], conf["stop"]
    dt = md["deltat"] * FS
    if md["type"].upper() == "NVT":
        T = md["temperature"]
        gamma = md["gamma"]
    crd = initx
    vel = initv
    e, f = - grad(atom, crd, nstep=-1)
    massm = genMassMat(atom) * AMU
    for nstep in range(md["nsteps"]):
        # velocity verlet
        crd = crd + vel * dt + 0.5 * (f / massm) * dt ** 2
        f_old = f
        e, f = - grad(atom, crd, nstep=nstep)
        print(">>> step: %i    e:%10.4f" % (nstep, e / EH))
        if md["type"].upper() == "NVT":
            f = f - gamma * vel + \
                np.sqrt(2. * gamma * KB * T) * \
                np.random.normal(0.0, np.ones(crd.shape))
        vel = vel + 0.5 * (f_old + f) / massm * dt
        # print
        if "freq" in prt and nstep % prt["freq"] == 0:
            if prt["coordinate"]:
                writeXYZ("%s-traj.xyz" % NAME, crd / ANGSTROM,
                         title="E:%10.6f NSTEP:%i" % (e, nstep), append=True)
            if prt["coordinate"]:
                writeXYZ("%s-vel.xyz" % NAME, vel / (ANGSTROM / FS),
                         title="E:%10.6f NSTEP:%i" % (e, nstep), append=True)
        # check_traj
        if "time" in chk and nstep == chk["time"]:
            for cv in chk["time"]["cv"]:
                if cv["type"].append() == "b":
                    r = distance(crd / ANGSTROM,
                                 cv["index"][0], cv["index"][1])
                    if r < cv["range"][0] or r > cv["range"][1]:
                        print(">>> Bond %i-%i out of range. Stop." %
                              (cv["index"][0], cv["index"][1]))
        # check_stop


def main():
    """
    The main function.
    """
    printTitle()
    conf = argparse()

    global NAME
    NAME = conf["name"]

    # build template for qm engine
    template = readFile(conf["force"]["template"],
                        lambda x: Template("".join(x)))

    grad = genGrad(conf["force"], template)
    # select init crd and vel
    atom, crd, vel = setInitMotion(conf["init"])
    # run dynamics
    dynamics(atom, crd, vel, grad=grad, conf=conf)
    print(">>> STDSP is finished.")


if __name__ == '__main__':
    main()
