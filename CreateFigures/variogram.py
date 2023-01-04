import os

import numpy as np
import skgstat as skg

from matplotlib import pyplot as plt
from netCDF4 import Dataset


def create_variogram():
    pass

def main():
    # Define paths
    PATH_DATA = "./local_data/AROME_1kmgrid_20220101T18Z.nc"

    with Dataset(PATH_DATA, 'r') as nc:
        x = nc.variables['x'][:1792:16]
        y = nc.variables['y'][578::16]
        t2m = nc.variables['t2m'][:,578::16,:1792:16]

    print(x.shape)
    print(y.shape)
    print(t2m.shape)

    xx, yy = np.meshgrid(x,y)
    coords = np.stack((xx.flatten(),yy.flatten()), axis = -1)
    
    print(coords.shape)
    print(t2m.shape)
    V = skg.Variogram(coords, t2m[0].flatten(), maxlag='meadian', n_lags = 15, normalize = 15)
    fig = V.plot(show = True)

    print('Done')


if __name__ == "__main__":
    main()