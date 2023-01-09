import h5py
import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from datetime import datetime, timedelta
from common_functions import onehot_encode_sic

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main():
    # Define paths
    path_nextsim = "/lustre/storeB/users/maltem/nowwind/cmems_mod_arc_phy_anfc_nextsim_hm_202007/"

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/"

    # This will be used to define the xy-boundary
    path_ml = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/weights_05011118/2022/01/"

    # Define projection transformer
    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_NEXTSIM = CRS.from_proj4(proj4_nextsim)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_NEXTSIM, always_xy = True)

    ml_forecast = (sorted(glob.glob(f"{path_ml}*")))[0]
    
    with h5py.File(ml_forecast, 'r') as infile:
        x_in = infile['xc'][:]
        y_in = infile['yc'][:]


    x_diff = x_in[1] - x_in[0]
    y_diff = y_in[1] - y_in[0]

    xc = np.pad(x_in, (1,1), 'constant', constant_values = (x_in[0] - x_diff, x_in[-1] + x_diff))
    yc = np.pad(y_in, (1,1), 'constant', constant_values = (y_in[0] - y_diff, y_in[-1] + y_diff))

    xxc, yyc = np.meshgrid(xc, yc)

    xxc_target, yyc_target = transform_function.transform(xxc, yyc)
    xxc_target_flat = xxc_target.flatten()
    yyc_target_flat = yyc_target.flatten()


    # Define months for parallel execution
    paths = []
    year = 2022
    for month in range(1, 13):
        p = f"{path_nextsim}{year}/{month:02d}/"
        paths.append(p)

    path_data_task = paths[int(sys.argv[1]) - 1]
    print(f"{path_data_task=}")
    path_output_task = path_data_task.replace(path_nextsim, path_output)
    print(f"path_output_task = {path_output_task}")
    year_task = path_data_task[len(path_nextsim):len(path_nextsim) + 4]
    print(f"{year_task=}")
    month_task = path_data_task[len(path_nextsim) + 5:len(path_nextsim) + 7]
    print(f"{month_task=}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"{nb_days_task=}")

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)


    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')

        lead_time_list = []
        

        yyyymmdd_nextsim_valid = (yyyymmdd_datetime + timedelta(days = 0)).strftime('%Y%m%d')
        try:
            nextsim_path = glob.glob(f"{path_nextsim}{year_task}/{month_task}/{yyyymmdd_nextsim_valid}_hr-nersc-MODEL-nextsimf-ARC-b{yyyymmdd}-fv00.0.nc")[0]

        except IndexError:
            print(f'Missing file at {yyyymmdd_nextsim_valid}-b{yyyymmdd}')
            continue

        with Dataset(nextsim_path, 'r') as nc:
            nextsim_x = nc.variables['x'][:]
            nextsim_y = nc.variables['y'][:]
            nextsim_lat = nc.variables['latitude'][:]         
            nextsim_lon = nc.variables['longitude'][:]
            nextsim_sic = nc.variables['siconc'][:]
            fill_value = nc.variables['siconc']._FillValue

        # The boundaries are defined as inclusive:exclusive
        leftmost_boundary = find_nearest(nextsim_x, np.min(xxc_target_flat))
        rightmost_boundary = find_nearest(nextsim_x, np.max(xxc_target_flat)) + 1

        lower_boundary = find_nearest(nextsim_y, np.min(yyc_target_flat))
        upper_boundary = find_nearest(nextsim_y, np.max(yyc_target_flat)) + 1

        nextsim_sic_current = np.ma.filled(nextsim_sic[:, lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary], fill_value = fill_value)

        lead_time_list.append(np.mean(nextsim_sic_current, axis = 0))
        
        nextsim_lsmask = np.where(lead_time_list[0] == fill_value, 1, 0)

        # yyyymmdd_nextsim_bulletin = (yyyymmdd_datetime + timedelta(days = 1)).strftime('%Y%m%d')
        for i in range(1, 3):
            yyyymmdd_nextsim_valid = (yyyymmdd_datetime + timedelta(days = i)).strftime('%Y%m%d')

            try:
                nextsim_path = glob.glob(f"{path_nextsim}{yyyymmdd_nextsim_valid[:4]}/{yyyymmdd_nextsim_valid[4:6]}/{yyyymmdd_nextsim_valid}_hr-nersc-MODEL-nextsimf-ARC-b{yyyymmdd}-fv00.0.nc")[0]

            except IndexError:
                print(f'Missing file at {yyyymmdd_nextsim_valid}-b{yyyymmdd}')
                continue

            with Dataset(nextsim_path, 'r') as nc:
                nextsim_sic_current = nc.variables['siconc'][:, lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary]
            
            lead_time_list.append(np.mean(np.ma.filled(nextsim_sic_current, fill_value), axis = 0))

        lead_time_array = np.array(lead_time_list)
        output_filename = f"nextsim_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(nextsim_x[leftmost_boundary:rightmost_boundary]))
            nc_out.createDimension('y', len(nextsim_y[lower_boundary:upper_boundary]))
            nc_out.createDimension('t', 3)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = nextsim_y[lower_boundary:upper_boundary]
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = nextsim_x[leftmost_boundary:rightmost_boundary]

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = nextsim_lat[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary]

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Lonitude'
            lonc[:] = nextsim_lon[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary]

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic(lead_time_array)

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = nextsim_lsmask

if __name__ == "__main__":
    main()