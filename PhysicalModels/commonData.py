import glob
import os

import numpy as np

from netCDF4 import Dataset


def main():
    
    # Define paths
    barents_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/barents/2022/01/barents_mean_b20220102.nc"
    ml_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/ml/lead_time_2/osisaf_trend_5/weights_05011118/2022/01/20220105_b20220103.nc"
    nextsim_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/2022/01/nextsim_mean_b20220101.nc"
    osisaf_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/osisaf/osisaf_trend_5/2022/01/osisaf_mean_b20220101.nc"

    path_target_nextsim = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/2022/01/nextsim_mean_b20220101.nc"

    with Dataset(path_target_nextsim, 'r') as nc_ns:
        nextsim_x = nc_ns['x'][:]
        nextsim_y = nc_ns['y'][:]

    nx = len(nextsim_x)
    ny = len(nextsim_y)

    with Dataset(barents_path, 'r') as nc:
        lsmask_barents = nc.variables['lsmask'][:]

    with Dataset(ml_path, 'r') as nc:
        lsmask_ml = nc.variables['lsmask'][:]

    with Dataset(nextsim_path, 'r') as nc:
        lsmask_nextsim = nc.variables['lsmask'][:]
        nextsim_lat = nc.variables['lat'][:]
        nextsim_lon = nc.variables['lon'][:]
        
    with Dataset(osisaf_path, 'r') as nc:
        lsmask_osisaf = nc.variables['lsmask'][:]

    # lsmask_merged = lsmask_barents + lsmask_ml + lsmask_nextsim + lsmask_osisaf
    lsmask_merged = lsmask_barents + lsmask_ml + lsmask_nextsim

    lsmask_merged[lsmask_merged > 1] = 1

    output_path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/"


    with Dataset(f"{output_path}commons.nc", 'w', format = "NETCDF4") as nc_out:
        nc_out.createDimension('x', nx)
        nc_out.createDimension('y', ny)

        lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
        lsmask_out.units = "1"
        lsmask_out.standard_name = "Land Sea Mask"
        lsmask_out[:] = lsmask_merged

        latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
        latc.units = 'degrees North'
        latc.standard_name = 'Latitude'
        latc[:] = nextsim_lat

        lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
        lonc.units = 'degrees East'
        lonc.standard_name = 'Lonitude'
        lonc[:] = nextsim_lon



if __name__ == "__main__":
    main()