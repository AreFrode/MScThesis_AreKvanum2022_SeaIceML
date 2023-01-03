# Script to regrid AA onto 1km grid
# Author: Are Frode Kvanum
# Date: 02.01.2023

import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid')


import glob
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from rotate_winds import rotate
from interpolate import nearest_neighbor_interp


def main():
    ################################################
	# Constants
	################################################
    path_data = '/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/'
    path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/'

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    ################################################
	# IceChart (AromeArctic 1km x 1km) grid
	################################################
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4

    nx = 1845
    ny = 2370

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    ################################################
    # Dataset
    ################################################
    paths = []
    for year in range(2019, 2023):
        for month in range(1, 13):
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)

    #
    # path_data_task = paths[$SGE_TASK_ID - 1]
    path_data_task = paths[int(sys.argv[1]) - 1] # This should be the only path
    print(f"path_data_task = {path_data_task}")
    path_output_task = path_data_task.replace(path_data, path_output)
    print(f"path_output_task = {path_output_task}")
    year_task = path_data_task[len(path_data) : len(path_data) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_data) + 5 : len(path_data) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")
    #
    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    ################################################
    # Data processing
    ################################################
    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        path_day = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_full_2_5km_{yyyymmdd}T18Z.nc")
        path_day_det = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_det_2_5km_{yyyymmdd}T18Z.nc")

        try:
            dataset = path_day[0]

        except IndexError:
            try:
                dataset = path_day_det[0]
            
            except IndexError:
                continue

        # Fetch variables from Arome Arctic
        with Dataset(dataset, 'r') as nc:

            x_input = nc.variables['x'][:]
            y_input = nc.variables['y'][:]
        
            lat_arome = nc.variables['latitude'][:]
            lon_arome = nc.variables['longitude'][:]
            t2m_arome = nc.variables['air_temperature_2m'][:]
            uwind_arome = nc.variables['x_wind_10m'][:]
            vwind_arome = nc.variables['y_wind_10m'][:]
            lsmask_arome = nc.variables['land_area_fraction'][:]

        nx_input = len(x_input)
        ny_input = len(y_input)
        xx_input, yy_input = np.meshgrid(x_input, y_input)


        # Allocate target arrays
        lat_target = np.zeros((ny, nx))
        lon_target = np.zeros((ny, nx))
        lsmask_target = np.zeros((ny, nx))
        t2m_target = np.zeros((3, ny, nx))
        xwind_target = np.zeros((3, ny, nx))
        ywind_target = np.zeros((3, ny, nx))

        t2m_cum_mean = np.zeros((3, ny_input, nx_input))
        uwind_cum_mean = np.zeros((3, ny_input, nx_input))
        vwind_cum_mean = np.zeros((3, ny_input, nx_input))


        # Compute "Cumulative" means for the temporal variables
        lead_times = [18, 42, 66]

        for idx, i in enumerate(lead_times):
            t2m_cum_mean[idx] = t2m_arome[0:i,...].mean(axis = 0)
            uwind_cum_mean[idx] = uwind_arome[0:i,0,...].mean(axis = 0)
            vwind_cum_mean[idx] = vwind_arome[0:i,0,...].mean(axis = 0)
        
        
        cat_fields = np.concatenate((lat_arome[None, :], lon_arome[None, :], lsmask_arome[0], t2m_cum_mean, uwind_cum_mean, vwind_cum_mean))

        regrid_cat = nearest_neighbor_interp(xx_input, yy_input, x_target, y_target, cat_fields)
        

        lat_target = regrid_cat[0]
        lon_target = regrid_cat[1]
        lsmask_target = regrid_cat[2]
        t2m_target = regrid_cat[3:6]

        uwind_regrid = regrid_cat[6:9]
        vwind_regrid = regrid_cat[9:12]

        # ROTATION assuming u: zonal, v: meridional
        for i in range(3):
            xwind_rotated, ywind_rotated = rotate(uwind_regrid[i].flatten(), vwind_regrid[i].flatten(), lat_target.flatten(), lon_target.flatten(), 'proj+=longlat', proj4_arome)
            xwind_target[i] = xwind_rotated.reshape(*xwind_target[i].shape)
            ywind_target[i] = ywind_rotated.reshape(*ywind_target[i].shape)


        ################################################
        # Output netcdf file
        ################################################
        output_filename = f"AROME_1kmgrid_{yyyymmdd}T18Z.nc"
        output_netcdf = Dataset(f"{path_output_task}{output_filename}", 'w', format = 'NETCDF4')

        output_netcdf.createDimension('y', len(y_target))
        output_netcdf.createDimension('x', len(x_target))
        output_netcdf.createDimension('t', 3)

        yc = output_netcdf.createVariable('y', 'd', ('y'))
        yc.units = 'm'
        yc.standard_name = 'y'

        xc = output_netcdf.createVariable('x', 'd', ('x'))
        xc.units = 'm'
        xc.standard_name = 'x'

        tc = output_netcdf.createVariable('t', 'd', ('t'))
        tc.units = 'days since forecast inception'
        tc.standard_name = 'time'

        latc = output_netcdf.createVariable('lat', 'd', ('y', 'x'))
        latc.units = 'degrees_north'
        latc.standard_name = 'Latitude'

        lonc = output_netcdf.createVariable('lon', 'd', ('y', 'x'))
        lonc.units = 'degrees_east'
        lonc.standard_name = 'Longitude'

        t2m_out = output_netcdf.createVariable('t2m', 'd', ('t', 'y', 'x'))
        t2m_out.units = 'K'
        t2m_out.standard_name = 'Air temperature at 2 metre height'

        xwind_out = output_netcdf.createVariable('xwind', 'd', ('t', 'y', 'x'))
        xwind_out.units = 'm/s'
        xwind_out.standard_name = 'x 10 metre wind (X10M)'

        ywind_out = output_netcdf.createVariable('ywind', 'd', ('t', 'y', 'x'))
        ywind_out.units = 'm/s'
        ywind_out.standard_name = 'y 10 metre wind (Y10M)'

        lsmask_out = output_netcdf.createVariable('lsmask', 'l', ('y', 'x'))
        lsmask_out.units = '1'
        lsmask_out .standard_name = 'Land Area Fraction'

        yc[:] = y_target
        xc[:] = x_target
        tc[:] = np.linspace(0,2,3)
        latc[:] = lat_target
        lonc[:] = lon_target
        lsmask_out[:] = lsmask_target
        t2m_out[:] = t2m_target
        xwind_out[:] = xwind_target
        ywind_out[:] = ywind_target

        ##
        output_netcdf.description = proj4_arome

        output_netcdf.close()



if __name__ == "__main__":
    main()