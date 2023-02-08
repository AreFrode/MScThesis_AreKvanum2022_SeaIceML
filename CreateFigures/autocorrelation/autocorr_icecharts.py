import glob

import seaborn as sns
import pandas as pd
import numpy as np

from netCDF4 import Dataset
from matplotlib import pyplot as plt

def autocorrelate(Data):
    """
    Modified from
    https://stackoverflow.com/questions/4503325/autocorrelation-of-a-multidimensional-array-in-numpy
    and
    https://en.wikipedia.org/wiki/Autocorrelation
    """
    Data = np.array(Data)
    A = Data.mean(axis=0)                        # Average 2D array
    New_Data = (Data - np.mean(A)) # Sample mean
    std = np.std(A)                # sample std -> Biased estimator
    n = len(New_Data)
    for count in range(0, 32):
        i = np.arange(n - count)
        yield np.divide(np.multiply(New_Data[i], New_Data[i+count]).sum(axis=0),(n - count)*std).mean()



def create_data():
    icecharts = sorted(glob.glob("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/**/*.nc"))

    Data = []
    for icechart in icecharts:
        with Dataset(icechart, 'r') as nc:
            Data.append(nc.variables['sic'][:])

    solution = np.array(list(autocorrelate(Data)))
    solution /= solution[0]

    with open('solution.npy', 'wb') as f:
        np.save(f, solution)

def plot_data():
    with open('solution.npy', 'rb') as f:
        solution = np.load(f)

    sns.set_theme()
    fig = plt.figure(figsize = (10, 6))
    ax = plt.axes()
    
    ax.plot(solution)
    ax.set_xlabel('Time lag [days]')
    ax.set_ylabel('Normalized mean autocorrelation')
    ax.set_title('Autocorrelation of sea ice charts')

    plt.savefig('autocorr_icechart.png')

if __name__ == "__main__":
    # create_data()
    plot_data()