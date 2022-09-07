import os
import glob

import numpy as np

from netCDF4 import Dataset


def get_invalid_coordinates():
    path_icechart = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/testingdata/2019/01/'

    nc = Dataset(glob.glob(f"{path_icechart}*.nc")[0], 'r')
    lat = nc.variables['lat'][:]

    return np.where(np.isnan(lat))

if __name__ == "__main__":
    get_invalid_coordinates()