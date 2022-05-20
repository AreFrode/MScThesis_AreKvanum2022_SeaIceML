import os

import numpy as np

from scipy.special import softmax
from matplotlib import pyplot as plt


def runstuff():
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/figures/"
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/outputs/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    data = np.loadtxt(f"{PATH_OUTPUT}pred_bins.txt", delimiter=',')

    targets = np.loadtxt(f"{PATH_OUTPUT}target_bins.txt", delimiter=',')

    SIC_field = np.empty(62500)
    SIC_target = np.empty(62500)

    for i in range(len(data)):
        # data[i] = softmax(data[i])

        SIC_field[i] = np.argmax(data[i])
        SIC_target[i] = np.argmax(targets[i])
        
    SIC_field = np.resize(SIC_field, (250,250))
    SIC_target = np.resize(SIC_target, (250,250))
    
    plt.figure()
    plt.pcolormesh(SIC_field)
    plt.colorbar()
    plt.savefig(f"{PATH_FIGURES}pred_bins.png")

    plt.figure()
    plt.pcolormesh(SIC_target)
    plt.colorbar()
    plt.savefig(f"{PATH_FIGURES}target_bins.png")





if __name__ == "__main__":
    runstuff()