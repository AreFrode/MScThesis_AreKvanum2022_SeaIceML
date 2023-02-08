import glob
import os
import h5py
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from netCDF4 import Dataset

from interpolate import nearest_neighbor_interp
from common_functions import onehot_encode_sic_numerical, get_target_domain
from datetime import datetime, timedelta

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():

    path_persistence = f"/lustre/storeB/users/nicholsh/icecharts_2022.nc"

    
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    # Set boundaries from ml domain


    # Define projection transformer
    common_grid = sys.argv[2]
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_arome, 'persistence')

    nx = len(target_x)
    ny = len(target_y)

    # Define months for parallel execution
    year = 2022
    months = []
    for month in range(1, 13):
        months.append(month)

    month_task = months[int(sys.argv[1]) - 1]
    print(f"{month_task=}")

    path_output_task = f"{path_output}{year}/{month_task:02d}/"
    print(f"{path_output_task=}")
    

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    with Dataset(f"{path_arome}2022/01/AROME_1kmgrid_20220101T18Z.nc") as constants:
        lsmask = constants['lsmask'][:,:]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)), 1, 0)
    

    nc = Dataset(path_persistence, 'r')
    persistence_x = nc.variables['x'][:]
    persistence_y = nc.variables['y'][:]
    persistence_time = nc.variables['time'][:]

    x_diff = persistence_x[1] - persistence_x[0]
    y_diff = persistence_y[1] - persistence_y[0]

    xc = np.pad(persistence_x, (1,1), 'constant', constant_values = (persistence_x[0] - x_diff, persistence_x[-1] + x_diff))
    yc = np.pad(persistence_y, (1,1), 'constant', constant_values = (persistence_y[0] - y_diff, persistence_y[-1] + y_diff))

    xxc, yyc = np.meshgrid(xc, yc)

    xxc_target, yyc_target = transform_function.transform(xxc, yyc)

    # Define datetime
    t0 = datetime(1981, 1, 1)

    for i in range(len(persistence_time)):
        time = t0 + timedelta(seconds=int(persistence_time[i]))
        current_year = time.year
        current_month = time.month
        current_day = time.day

        if current_year != year or current_month != month_task:
            continue
        
        yyyymmdd = f"{current_year}{current_month:02d}{current_day:02}"

        path_output_task = f"{path_output}{current_year}/{current_month:02d}/"
        # print(f"path_output_task = {path_output_task}")
        
        if not os.path.exists(path_output_task):
            os.makedirs(path_output_task)

        persistence_sic = nc.variables['sic'][i, :]
        persistence_sic = np.where(mask == 0, -1, persistence_sic)

        persistence_sic_padded = np.pad(persistence_sic, ((1,1), (1,1)), 'constant', constant_values = np.nan)

        interp_target = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, persistence_sic_padded, fill_value = -1)



        output_filename = f"target_v{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', nx)
            nc_out.createDimension('y', ny)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = target_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = target_x

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = target_lat

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Lonitude'
            lonc[:] = target_lon

            sic_out = nc_out.createVariable('sic', 'd', ('y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(interp_target)

    nc.close()

if __name__ == "__main__":
    main()