import os

import numpy as np
import gstools as gs
import skgstat as skg
import seaborn as sns

from matplotlib import pyplot as plt
from netCDF4 import Dataset


def create_variogram():
    pass

def main():
    # Define paths
    PATH_DATA = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/local_data/AROME_1kmgrid_20220101T18Z.nc"

    sample_rate = 1
    t = 0
    with Dataset(PATH_DATA, 'r') as nc:
        lat = nc.variables['lat'][578::sample_rate, :1792:sample_rate]
        lon = nc.variables['lon'][578::sample_rate, :1792:sample_rate]
        x = nc.variables['x'][:1792:sample_rate]
        y = nc.variables['y'][578::sample_rate]
        t2m = nc.variables['t2m'][t,578::sample_rate,:1792:sample_rate]
        xwind = nc.variables['xwind'][t,578::sample_rate,:1792:sample_rate]
        ywind = nc.variables['ywind'][t,578::sample_rate,:1792:sample_rate]

    print(x.shape)
    print(y.shape)
    print(t2m.shape)

    print(np.var(t2m))

 
    # fig = plt.figure()
    # plt.pcolormesh(y, x, t2m, shading='auto')
    # plt.show()


    xx, yy = np.meshgrid(x,y)
    # coords = np.stack((xx.flatten(),yy.flatten()), axis = -1)
    coords = np.stack((lat.flatten(),lon.flatten()))

    # V = gs.vario_estimate(coords, t2m.flatten(), latlon=True)

    # field = t2m[0].flatten()
    
    # print(coords.shape)
    # print(t2m.shape)
    # V = skg.Variogram(coords, t2m.flatten(), maxlag='median', n_lags = 25, normalize = False, model='gaussian')
    # fig = V.plot()
    # fig.savefig('Variogram_t2m.png')



    # bin_center, gamma = gs.vario_estimate((xx.flatten(), yy.flatten()), t2m[0].flatten())
    # print(bin_center)
    # print(gamma)
    
    # V = skg.Variogram(coords, t2m.flatten(), estimator='matheron', bin_func='even', # model='gaussian', maxlag = '3000')
    # bin_center, gamma = V.get_empirical(bin_center = True)

    # print(bin_center)
    # print(gamma)

    # sns.set_theme()

    # print(V)
    # sill = V.parameters[1]

    # print(sill - gamma)
    
    # fig = V.plot(show = False)

    # axes = fig.axes

    # axes[0].set_xlabel('Lag (-) [1e6 km]')
    # fig.savefig(f'Variogram_t2m.png')

    exit()

    fields = np.stack((xwind, ywind))
    for name, field, direction in zip(['xwind', 'ywind'], fields, [0, 90]):
        V = skg.DirectionalVariogram(coords, field.flatten(), azimuth = direction, direction = 90, n_lags = 21, estimator='matheron', bin_func='even', normalize=True, model='gaussian')
        bin_center, gamma = V.get_empirical(bin_center = True)

        print(bin_center)
        print(gamma)

        # sns.set_theme()

        print(V)
    
        fig = V.plot(show = False)

        axes = fig.axes

        # axes[0].set_xlabel('Lag (-)')
        fig.savefig(f'Variogram_{name}.png')

        # fig2 = V.distance_difference_plot()
        # fig2.savefig(f'Distance_{i}.png')

    print('Done')


if __name__ == "__main__":
    main()
