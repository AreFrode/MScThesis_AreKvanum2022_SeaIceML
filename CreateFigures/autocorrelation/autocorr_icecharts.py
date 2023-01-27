import glob

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
    for count in range(0, 31):
        i = np.arange(n - count)
        yield np.divide(np.multiply(New_Data[i], New_Data[i+count]).sum(axis=0),(n - count)*std).mean()



icecharts = sorted(glob.glob("/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/local_data/2022/**/*.nc"))

Data = []
for icechart in icecharts:
    with Dataset(icechart, 'r') as nc:
        Data.append(nc.variables['sic'][:])

solution = np.array(list(autocorrelate(Data)))
solution /= solution[0]

with open ('solution.npy', 'wb') as f:
    np.save(f, solution)