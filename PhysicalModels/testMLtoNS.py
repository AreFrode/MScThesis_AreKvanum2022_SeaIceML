import h5py
import glob
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    path_nextsim = "/lustre/storeB/users/maltem/nowwind/cmems_mod_arc_phy_anfc_nextsim_hm_202007/2022/01/"
    path_ml = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/weights_13121158/2021/01/"

    # Define transformer
    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_NEXTSIM = CRS.from_proj4(proj4_nextsim)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_NEXTSIM, always_xy = True)

    ml_forecast = (sorted(glob.glob(f"{path_ml}*")))[0]
    nextsim_forecast = (sorted(glob.glob(f'{path_nextsim}*')))[0]
    
    with h5py.File(ml_forecast, 'r') as infile:
        x_in = infile['xc'][:]
        y_in = infile['yc'][:]
        ml_sic = infile['y_pred'][0,:,:]


    x_diff = x_in[1] - x_in[0]
    y_diff = y_in[1] - y_in[0]

    xc = np.pad(x_in, (1,1), 'constant', constant_values = (x_in[0] - x_diff, x_in[-1] + x_diff))
    yc = np.pad(y_in, (1,1), 'constant', constant_values = (y_in[0] - y_diff, y_in[-1] + y_diff))

    ml_sic_padded = np.pad(ml_sic, ((1,1), (1,1)), 'constant', constant_values = -1)

    xxc, yyc = np.meshgrid(xc, yc)

    plt.figure()
    plt.pcolormesh(xxc, yyc, ml_sic_padded, shading = 'auto')
    plt.savefig('ml_original_pcolor.png')

    xxc_target, yyc_target = transform_function.transform(xxc, yyc)
    xxc_target_flat = xxc_target.flatten()
    yyc_target_flat = yyc_target.flatten()

    with Dataset(nextsim_forecast, 'r') as nc:
        nextsim_x = nc.variables['x'][:]
        nextsim_y = nc.variables['y'][:]
        nextsim_sic = nc.variables['siconc'][:]


        # The boundaries are defined as inclusive:exclusive
        leftmost_boundary = find_nearest(nextsim_x, np.min(xxc_target_flat))
        rightmost_boundary = find_nearest(nextsim_x, np.max(xxc_target_flat)) + 1

        lower_boundary = find_nearest(nextsim_y, np.min(yyc_target_flat))
        upper_boundary = find_nearest(nextsim_y, np.max(yyc_target_flat)) + 1

    x_target = nextsim_x[leftmost_boundary:rightmost_boundary]
    y_target = nextsim_y[lower_boundary:upper_boundary]

    ml_sic_flat = ml_sic_padded.flatten()

    ml_sic_target = griddata((xxc_target_flat, yyc_target_flat), ml_sic_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

    xx_target, yy_target = np.meshgrid(x_target, y_target)

    plt.figure()
    plt.pcolormesh(xx_target, yy_target, ml_sic_target, shading = 'auto')
    plt.savefig('ml_test_pcolor.png')
    
    plt.figure()
    plt.pcolormesh(xx_target, yy_target, nextsim_sic[0, lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary], shading = 'auto')
    plt.savefig('test_pcolor.png')



if __name__ == "__main__":
    main()