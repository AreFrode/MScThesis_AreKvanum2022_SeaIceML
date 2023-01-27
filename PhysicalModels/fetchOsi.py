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
from Regrid_OsiSaf import compute_trend_1d, mask_land
from common_functions import onehot_encode_sic_numerical, get_target_domain

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths

    path_osi = f"/lustre/storeB/project/copernicus/osisaf/data/archive/ice/conc/"
    common_grid = sys.argv[2]

    proj4_osi = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45"

    # Define trend length
    num_days = [3, 5, 7]
    n_trends = len(num_days)

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_osi, 'osisaf')

    nx = len(target_x)
    ny = len(target_y)

    xc_osisaf = 760
    yc_osisaf = 1120

    # Prepare masks
    baltic_mask = np.zeros((yc_osisaf, xc_osisaf))
    baltic_mask[610:780, 625:700] = 1
    
    # Define months for parallel execution
    year = 2022
    paths = []
    for month in range(1, 13):
        p = f"{path_osi}{year}/{month:02d}/"
        paths.append(p)

    path_data_task = paths[int(sys.argv[1]) - 1] # This should be the only path
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
        
        path_osisaf_task = glob.glob(f"{path_data_task}/ice_conc_nh_polstere-100_multi_{yyyymmdd}1200.nc")

        
        # Read input
        raw_ice_conc = np.empty((num_days[-1], yc_osisaf, xc_osisaf))
        raw_ice_conc.fill(np.nan)


        try:
            dataset = path_osisaf_task[0]

        except IndexError:
            continue

        with Dataset(dataset, 'r') as nc:
            x_input = nc.variables['xc'][:] * 1000
            y_input = nc.variables['yc'][:] * 1000

            fill_value = nc.variables['ice_conc']._FillValue

            ice_conc = nc.variables['ice_conc'][0,:,:]
            lsmask = np.where(ice_conc == fill_value, 1, 0)

            tmp_ice_conc = np.ma.filled(ice_conc, fill_value = fill_value)
            tmp_ice_conc = np.where(baltic_mask == 1, fill_value, tmp_ice_conc)

            raw_ice_conc[0] = mask_land(tmp_ice_conc, fill_value)
            
        

        for i in range(1, num_days[-1]):
            yyyymmdd_current = (yyyymmdd_datetime - timedelta(days = i)).strftime('%Y%m%d')

            try:
                path_current = glob.glob(f"{path_osi}{yyyymmdd_current[:4]}/{yyyymmdd_current[4:6]}/ice_conc_nh_polstere-100_multi_{yyyymmdd_current}1200.nc")[0]
            
            # If missing days, compute trend from remainder of days
            except IndexError:
                continue
                
            with Dataset(path_current, 'r') as nc:
                tmp_ice_conc = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = fill_value)
                tmp_ice_conc = np.where(baltic_mask == 1, fill_value, tmp_ice_conc)


                raw_ice_conc[i] = mask_land(tmp_ice_conc, fill_value)

        trend_array = np.zeros((n_trends, yc_osisaf, xc_osisaf))

        for i in range(n_trends):
            valid_length = [~np.isnan(raw_ice_conc[j, :, :]).all() for j in range(num_days[i])]
            valid_days = np.sum(valid_length)

            if valid_days > 1:
                trend_array[i] = np.apply_along_axis(compute_trend_1d, axis = 0, arr  = raw_ice_conc[:num_days[i],:,:])

            else:
                trend_array[i] = fill_value

            
        xxc, yyc = np.meshgrid(x_input, y_input)
        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        interp_array = np.concatenate((np.expand_dims(lsmask, axis=0), np.expand_dims(raw_ice_conc[0], axis=0), trend_array))
        
        interp_target = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)

        ice_conc_days = np.zeros((n_trends, 3, *interp_target[1].shape))
        for i in range(n_trends):
            for j in range(3):
                ice_conc_days[i,j] = interp_target[1] + (j + 1) * interp_target[2 + i]
        
        ice_conc_days[ice_conc_days < 0] = 0
        ice_conc_days[ice_conc_days > 100] = 100

        output_filename = f"osisaf_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', nx)
            nc_out.createDimension('y', ny)
            nc_out.createDimension('memb', 3)
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

            sic_out = nc_out.createVariable('sic', 'd', ('memb', 't', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(ice_conc_days)

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interp_target[0]
            

if __name__ == "__main__":
    main()