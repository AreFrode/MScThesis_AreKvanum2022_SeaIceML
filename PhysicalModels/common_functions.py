import numpy as np
import h5py
import glob

from netCDF4 import Dataset
from pyproj import CRS, Transformer

def onehot_encode_sic(sic):
    fast_ice = np.where(np.equal(sic, 1.), 6, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, .9), np.less(sic,1.)), 5, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, .7), np.less(sic, .9)), 4, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, .4), np.less(sic, .7)), 3, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, .1), np.less(sic, .4)), 2, 0)
    open_water = np.where(np.logical_and(np.greater(sic, 0.), np.less(sic, .1)), 1, 0)
    invalid_above = np.where(np.logical_or(np.isnan(sic), np.greater(sic, 1.)), -10, 0)
    invalid_below = np.where(np.less(sic, 0.), -10, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice + open_water + invalid_above + invalid_below

def onehot_encode_sic_numerical(sic):
    fast_ice = np.where(np.equal(sic, 100.), 6, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, 90.), np.less(sic,100.)), 5, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, 70.), np.less(sic, 90.)), 4, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, 40.), np.less(sic, 70.)), 3, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, 10.), np.less(sic, 40.)), 2, 0)
    open_water = np.where(np.logical_and(np.greater(sic, 0.), np.less(sic, 10.)), 1, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice + open_water

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_ml_domain_borders(path_ml, transform_function):
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

    return xxc_target_flat, yyc_target_flat

def get_target_domain(common_grid, current_proj, product):
    if common_grid == 'nextsim':
        path_common_grid = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/2022/01/nextsim_mean_b20220101.nc"
        proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
        crs_TARGET= CRS.from_proj4(proj4_nextsim)


    elif common_grid == 'amsr2':
        path_common_grid = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2/2022/01/target_v20220101.nc"
        epsg_amsr2 = 3411
        crs_TARGET = CRS.from_epsg(epsg_amsr2)


    else:
        exit('No Common grid provided')

    # Set boundaries from ml domain
    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{common_grid}_grid/{product}/"


    # Define projection transformer
    
    crs_CURRENT = CRS.from_proj4(current_proj)
    transform_function = Transformer.from_crs(crs_CURRENT, crs_TARGET, always_xy = True)

    # Define target grid
    with Dataset(path_common_grid, 'r') as nc_t:
        target_x = nc_t['x'][:]
        target_y = nc_t['y'][:]
        target_lat = nc_t['lat'][:]
        target_lon = nc_t['lon'][:]

    return path_output, transform_function, target_x, target_y, target_lat, target_lon