import os

import numpy as np

from matplotlib import pyplot as plt


def runstuff():
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/figures/"
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/outputs/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    data = np.loadtxt(f"{PATH_OUTPUT}pred.txt")

    targets = np.loadtxt(f"{PATH_OUTPUT}target.txt")

    SIC_field = np.resize(data, (250,250))
    SIC_field = np.where(SIC_field > 0.5, 1, 0)
    
    plt.figure()
    plt.pcolormesh(SIC_field)
    plt.colorbar()
    plt.savefig(f"{PATH_FIGURES}pred.png")

    plt.figure()
    plt.pcolormesh(np.resize(targets, (250,250)))
    plt.colorbar()
    plt.savefig(f"{PATH_FIGURES}target.png")





if __name__ == "__main__":
    runstuff()