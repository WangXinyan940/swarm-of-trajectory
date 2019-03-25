import matplotlib.pyplot as plt 
import numpy as np
import json
import sys
import os
import sot

def argparse(args):
    for n,i in enumerate(args):
        if i == "-j":
            conf = args[n+1]
            with open(conf, "r") as f:
                c = json.loads("".join(f))
            return c
    raise BaseException("Configure file needed.")

def distance(vi, vj):
    return ((vi - vj) ** 2).sum() ** 0.5





def main():
    conf = argparse(sys.argv)
    traj_list = []
    cvs = conf["cv"]
    dof = len(cvs)
    for dt in conf["data"]:
        dirs = [i for i in os.listdir(dt["path"]) if os.path.isdir(dt["path"]+"/"+i)]
        for d in dirs:
            #print(dt["path"] + "/" + d + "/" + dt["name"])
            xyzs = sot.readFile(dt["path"]+"/"+d+"/"+dt["name"], sot.readMultiXYZ)
            if len(xyzs) < conf["minlength"]:
                continue
            trj = np.zeros((len(xyzs),dof))
            for ni,i in enumerate(xyzs):
                for nj,j in enumerate(cvs):
                    if j["type"].upper() == "B":
                        trj[ni,nj] = distance(i[1][j["index"][0]], i[1][j["index"][1]])
            traj_list.append(trj)
    print(">>> %i trajectories. Drawing picture"%len(traj_list))
    if len(cvs) == 2:
        plt.xlabel(cvs[0]["name"])
        plt.ylabel(cvs[1]["name"])
    for d in traj_list:
        plt.plot(d[:,0], d[:,1], c="black", alpha=0.4)
    for d in traj_list:
        plt.scatter(d[0,0], d[0,1], c="red", s=10)
    plt.savefig(conf["output"])


if __name__ == '__main__':
    main()