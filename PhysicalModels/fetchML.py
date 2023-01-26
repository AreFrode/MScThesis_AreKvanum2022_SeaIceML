import h5py
import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer, Proj
from scipy.interpolate import griddata
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from helper_functions import read_config_from_csv
from interpolate import nearest_neighbor_interp
from common_functions import get_target_domain

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    assert len(sys.argv) > 2, 'supply valid model weights and thread member'
    weights = sys.argv[1]

    # Read config csv
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    config = read_config_from_csv(f"{PATH_OUTPUTS}configs/{weights}.csv")

    # Set the forecast lead time and osi trend
    lead_time = int(config['lead_time'])

    common_grid = sys.argv[3]

    path_ml = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/{weights}/"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_arome, 'ml')

    nx = len(target_x)
    ny = len(target_y)

    
    # Define lsmask
    path_lsmask = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/01/"
    h5file = sorted(glob.glob(f"{path_lsmask}*.hdf5"))[0]
    with h5py.File(h5file, 'r') as f:
        lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

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

    files = sorted(glob.glob(f"{path_ml}{year}/{month_task:02d}/*.hdf5"))

    for file in files:
        yyyymmdd = file[-17:-9]
        
        print(yyyymmdd)
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_bulletin = (yyyymmdd_datetime - timedelta(days = lead_time)).strftime('%Y%m%d')

        with h5py.File(file, 'r') as infile:
            ml_x = infile['xc'][:]
            ml_y = infile['yc'][:]
            ml_sic = infile['y_pred'][0,:,:]

        x_diff = ml_x[1] - ml_x[0]
        y_diff = ml_y[1] - ml_y[0]

        xc = np.pad(ml_x, (1,1), 'constant', constant_values = (ml_x[0] - x_diff, ml_x[-1] + x_diff))
        yc = np.pad(ml_y, (1,1), 'constant', constant_values = (ml_y[0] - y_diff, ml_y[-1] + y_diff))

        lsmask_padded = np.pad(lsmask, ((1,1), (1,1)), 'constant', constant_values = 1)
        ml_sic_padded = np.pad(ml_sic, ((1,1), (1,1)), 'constant', constant_values = -1)

        xxc, yyc = np.meshgrid(xc, yc)

        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        interp_array = np.stack((lsmask_padded, ml_sic_padded))

        interp_target = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)

        output_filename = f"{weights}_{yyyymmdd}_b{yyyymmdd_bulletin}.nc"


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
            sic_out[:] = interp_target[1]

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interp_target[0]


if __name__ == "__main__":
    main()