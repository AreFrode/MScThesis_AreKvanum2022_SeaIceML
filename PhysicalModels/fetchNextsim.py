import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from datetime import datetime, timedelta
from common_functions import onehot_encode_sic, get_target_domain
from interpolate import nearest_neighbor_interp


def main():
    # Define paths
    path_nextsim = "/lustre/storeB/users/maltem/nowwind/cmems_mod_arc_phy_anfc_nextsim_hm_202007/"

    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain('amsr2', proj4_nextsim, 'nextsim')

    nx = len(target_x)
    ny = len(target_y)
    
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
            nextsim_sic = nc.variables['siconc'][:]
            fill_value = nc.variables['siconc']._FillValue

        xxc, yyc = np.meshgrid(nextsim_x, nextsim_y)
        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        nextsim_sic_current = np.ma.filled(nextsim_sic[:], fill_value = fill_value)

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
                nextsim_sic_current = nc.variables['siconc'][:]
            
            lead_time_list.append(np.mean(np.ma.filled(nextsim_sic_current, fill_value), axis = 0))

        lead_time_array = np.array(lead_time_list)

        interp_array = np.concatenate((np.expand_dims(nextsim_lsmask, axis = 0), lead_time_array), axis = 0)
        
        interpolated = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)


        output_filename = f"nextsim_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', nx)
            nc_out.createDimension('y', ny)
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

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic(interpolated[1:])

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interpolated[0]

if __name__ == "__main__":
    main()