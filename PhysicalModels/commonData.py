import glob
import os
import sys

import numpy as np

from netCDF4 import Dataset


def main():

    target_grid = sys.argv[1]

    if target_grid == 'nextsim':
        file = 'nextsim_mean_b20220101.nc'

    elif target_grid == 'amsr2':
        file = 'target_v20220101.nc'

    else:
        exit('No valid target grid')
    
    # Define paths
    barents_path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{target_grid}_grid/barents/2022/01/barents_mean_b20220102.nc"
    
    ml_path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{target_grid}_grid/ml/2022/01/weights_05011118_20220105_b20220103.nc"

    # This should append nextsim lsmask twice, not a bug just lazy
    nextsim_path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{target_grid}_grid/nextsim/2022/01/nextsim_mean_b20220101.nc"
    # osisaf_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/osisaf/osisaf_trend_5/2022/01/osisaf_mean_b20220101.nc"



    path_target = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{target_grid}/2022/01/{file}"

    lsmasks = []

    with Dataset(path_target, 'r') as nc_t:
        target_x = nc_t['x'][:]
        target_y = nc_t['y'][:]
        target_lat = nc_t['lat'][:]
        target_lon = nc_t['lon'][:]
        lsmasks.append(nc_t['lsmask'][:])

    nx = len(target_x)
    ny = len(target_y)

    with Dataset(barents_path, 'r') as nc:
        lsmasks.append(nc.variables['lsmask'][:])

    with Dataset(ml_path, 'r') as nc:
        lsmasks.append(nc.variables['lsmask'][:])

    with Dataset(nextsim_path, 'r') as nc:
        lsmasks.append(nc.variables['lsmask'][:])
        
    # with Dataset(osisaf_path, 'r') as nc:
        # lsmask_osisaf = nc.variables['lsmask'][:]

    lsmasks = np.array(lsmasks)
    lsmask_merged = np.sum(lsmasks, axis = 0)

    lsmask_merged[lsmask_merged > 1] = 1

    output_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/"


    with Dataset(f"{output_path}{target_grid}_commons.nc", 'w', format = "NETCDF4") as nc_out:
        nc_out.createDimension('x', nx)
        nc_out.createDimension('y', ny)

        lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
        lsmask_out.units = "1"
        lsmask_out.standard_name = "Land Sea Mask"
        lsmask_out[:] = lsmask_merged

        latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
        latc.units = 'degrees North'
        latc.standard_name = 'Latitude'
        latc[:] = target_lat

        lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
        lonc.units = 'degrees East'
        lonc.standard_name = 'Lonitude'
        lonc[:] = target_lon



if __name__ == "__main__":
    main()