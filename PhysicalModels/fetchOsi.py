import h5py
import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from interpolate import nearest_neighbor_interp
from Regrid_OsiSaf import compute_trend_1d
from common_functions import onehot_encode_sic_numerical

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths

    path_osi = f"/lustre/storeB/project/copernicus/osisaf/data/archive/ice/conc/"
    path_target_nextsim = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/2022/01/nextsim_mean_b20220101.nc"

    # Define trend length
    num_days = int(sys.argv[1])

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/osisaf/osisaf_trend_{num_days}/"

    # Define projection transformer
    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
    proj4_osisaf = "+proj=laea +a=6371228.0 +lat_0=90 +lon_0=0"

    crs_NEXTSIM = CRS.from_proj4(proj4_nextsim)
    crs_OSISAF= CRS.from_proj4(proj4_osisaf)
    transform_function = Transformer.from_crs(crs_OSISAF, crs_NEXTSIM, always_xy = True)

    # Define target grid
    with Dataset(path_target_nextsim, 'r') as nc_ns:
        nextsim_x = nc_ns['x'][:]
        nextsim_y = nc_ns['y'][:]
        nextsim_lat = nc_ns['lat'][:]
        nextsim_lon = nc_ns['lon'][:]

    nx = len(nextsim_x)
    ny = len(nextsim_y)
    
    # Define months for parallel execution
    year = 2022
    paths = []
    for month in range(1, 13):
        p = f"{path_osi}{year}/{month:02d}/"
        paths.append(p)

    path_data_task = paths[int(sys.argv[2]) - 1] # This should be the only path
    print(f"{path_data_task=}")
    path_output_task = path_data_task.replace(path_osi, path_output)
    print(f"path_output_task = {path_output_task}")
    year_task = path_data_task[len(path_osi):len(path_osi) + 4]
    print(f"{year_task=}")
    month_task = path_data_task[len(path_osi) + 5:len(path_osi) + 7]
    print(f"{month_task=}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"{nb_days_task=}")
    #
    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        
        path_osisaf_task = glob.glob(f"{path_data_task}/ice_conc_nh_ease-125_multi_{yyyymmdd}1200.nc")

        
        # Read input
        raw_ice_conc = np.empty((num_days, 849, 849))
        raw_ice_conc.fill(np.nan)


        try:
            dataset = path_osisaf_task[0]

        except IndexError:
            continue

        with Dataset(dataset, 'r') as nc:
            x_input = nc.variables['xc'][:] * 1000
            y_input = nc.variables['yc'][:] * 1000
            fill_value = nc.variables['ice_conc']._FillValue
            raw_ice_conc[0] = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = fill_value)
            
        lsmask = np.where(raw_ice_conc[0] == fill_value, 1, 0)

        for i in range(1, num_days):
            yyyymmdd_current = (yyyymmdd_datetime - timedelta(days = i)).strftime('%Y%m%d')

            try:
                path_current = glob.glob(f"{path_osi}{yyyymmdd_current[:4]}/{yyyymmdd_current[4:6]}/ice_conc_nh_ease-125_multi_{yyyymmdd_current}1200.nc")[0]
            
            # If missing days, compute trend from remainder of days
            except IndexError:
                continue
                
            with Dataset(path_current, 'r') as nc:
                raw_ice_conc[i] = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = fill_value)

        ice_conc_trend = np.apply_along_axis(compute_trend_1d, axis = 0, arr = raw_ice_conc)
            
        xxc, yyc = np.meshgrid(x_input, y_input)
        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        interp_array = np.stack((lsmask, raw_ice_conc[0], ice_conc_trend))
        
        interp_target = nearest_neighbor_interp(xxc_target, yyc_target, nextsim_x, nextsim_y, interp_array)

        ice_conc_days = np.zeros((3, *interp_target[1].shape))
        for i in range(3):
            ice_conc_days[i] = interp_target[1] + (i + 1) * interp_target[2]
        
        ice_conc_days[ice_conc_days < 0] = 0
        ice_conc_days[ice_conc_days > 100] = 100

        output_filename = f"osisaf_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', nx)
            nc_out.createDimension('y', ny)
            nc_out.createDimension('t', 3)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = nextsim_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = nextsim_x

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = nextsim_lat

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Lonitude'
            lonc[:] = nextsim_lon

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(ice_conc_days)

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interp_target[0]

if __name__ == "__main__":
    main()