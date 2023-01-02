import h5py
import glob
import os
import sys

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
    path_barents = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/eps/"
    path_target_nextsim = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/lead_time_02/2022/01/nextsim_mean_20220105_b20220104.nc"

    # Set the forecast lead time
    lead_time = int(sys.argv[1])

    # Set boundaries from ml domain

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/barents/lead_time_{lead_time:02d}/"

    # PrepareData will be used to keep track of bulletindates
    if lead_time == 1:
        path_raw = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/"
        start = 0
        stop = 24
    
    elif lead_time == 2:
        path_raw = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
        start = 24
        stop = 48

    elif lead_time == 3:
        path_raw = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/three_day_forecast/"
        start = 48
        stop = 66

    else:
        print('No valid lead time supplied')
        exit()

    # Define projection transformer
    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_NEXTSIM = CRS.from_proj4(proj4_nextsim)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_NEXTSIM, always_xy = True)


    # Define target grid
    with Dataset(path_target_nextsim, 'r') as nc_ns:
        nextsim_x = nc_ns['x'][:]
        nextsim_y = nc_ns['y'][:]

    nx = len(nextsim_x)
    ny = len(nextsim_y)

    # Define months for parallel execution
    year = 2022
    months = []
    for month in range(1, 13):
        months.append(month)

    month_task = months[int(sys.argv[2]) - 1]
    print(f"{month_task=}")

    path_output_task = f"{path_output}{year}/{month_task:02d}/"
    print(f"{path_output_task=}")
    
    nb_days_task = monthrange(int(year), int(month_task))[1]
    print(f"{nb_days_task=}")

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year}{month_task:02d}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_barents_bulletin = (yyyymmdd_datetime + timedelta(days = 1)).strftime('%Y%m%d')

        try:
            raw_path = glob.glob(f"{path_raw}{year}/{month_task:02d}/PreparedSample_{yyyymmdd}.hdf5")[0]
            barents_path = glob.glob(f"{path_barents}barents_eps_{yyyymmdd_barents_bulletin}T00Z.nc")[0]

        except IndexError:
            continue

        with Dataset(barents_path, 'r') as nc:
            barents_x = nc.variables['X'][:]
            barents_y = nc.variables['Y'][:]
            barents_sic = nc.variables['ice_concentration'][:,:,:,:]

        x_diff = barents_x[1] - barents_x[0]
        y_diff = barents_y[1] - barents_y[0]

        xc = np.pad(barents_x, (1,1), 'constant', constant_values = (barents_x[0] - x_diff, barents_x[-1] + x_diff))
        yc = np.pad(barents_y, (1,1), 'constant', constant_values = (barents_y[0] - y_diff, barents_y[-1] + y_diff))

        barents_sic_padded = np.pad(barents_sic, ((0,0), (0,0), (1,1), (1,1)), 'constant', constant_values = np.nan)

        xxc, yyc = np.meshgrid(xc, yc)

        xxc_target, yyc_target = transform_function.transform(xxc, yyc)
        xxc_target_flat = xxc_target.flatten()
        yyc_target_flat = yyc_target.flatten()

        # Allocate target arrays
        sic_target = np.zeros((6, ny, nx))

        for i in range(6):
            barents_sic_flat = np.mean(barents_sic_padded[start:stop, i, ...], axis=0).flatten()

            sic_target[i] = griddata((xxc_target_flat, yyc_target_flat), barents_sic_flat, (nextsim_x[None, :], nextsim_y[:, None]), method = 'nearest')

        output_filename = f"barents_mean_b{yyyymmdd_barents_bulletin}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(nextsim_x))
            nc_out.createDimension('y', len(nextsim_y))
            nc_out.createDimension('member', 6)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = nextsim_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = nextsim_x

            sic_out = nc_out.createVariable('sic', 'd', ('member', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = sic_target



if __name__ == "__main__":
    main()