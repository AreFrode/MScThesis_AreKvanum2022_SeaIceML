import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset

from interpolate import nearest_neighbor_interp
from common_functions import onehot_encode_sic, get_target_domain

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    # Use processed largest grid for regrid domain
    common_grid = 'amsr2'

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_arome, 'arome_winds')

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
    
    nb_days_task = monthrange(int(year), int(month_task))[1]
    print(f"{nb_days_task=}")

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year}{month_task:02d}{dd:02d}"
        print(yyyymmdd)

        try:
             arome_path = glob.glob(f"{path_arome}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/AROME_1kmgrid_{yyyymmdd}T18Z.nc")[0]

        except IndexError:
            print(f"Missing file b{yyyymmdd}")
            continue

        with Dataset(arome_path, 'r') as nc:
            arome_x = nc.variables['x'][:]
            arome_y = nc.variables['y'][:]
            xwind = nc.variables['xwind'][:]
            ywind = nc.variables['ywind'][:]

        x_diff = arome_x[1] - arome_x[0]
        y_diff = arome_y[1] - arome_y[0]

        xc = np.pad(arome_x, (1,1), 'constant', constant_values = (arome_x[0] - x_diff, arome_x[-1] + x_diff))
        yc = np.pad(arome_y, (1,1), 'constant', constant_values = (arome_y[0] - y_diff, arome_y[-1] + y_diff))

        xwind_padded = np.pad(xwind, ((0,0), (1,1), (1,1)), 'constant', constant_values = np.nan)
        ywind_padded = np.pad(ywind, ((0,0), (1,1), (1,1)), 'constant', constant_values = np.nan)


        xxc, yyc = np.meshgrid(xc, yc)

        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        interp_array = np.concatenate((xwind_padded, ywind_padded))

        interpolated = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)
        
        output_filename = f"arome_winds_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(target_x))
            nc_out.createDimension('y', len(target_y))
            nc_out.createDimension('t', 3)

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

            xwind_out = nc_out.createVariable('xwind', 'd', ('t', 'y', 'x'))
            xwind_out[:] = interpolated[:3]

            ywind_out = nc_out.createVariable('ywind', 'd', ('t', 'y', 'x'))
            ywind_out[:] = interpolated[3:]


if __name__ == "__main__":
    main()