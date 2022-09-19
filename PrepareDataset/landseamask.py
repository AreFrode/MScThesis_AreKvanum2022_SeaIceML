import glob
import os
import numpy as np

from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from netCDF4 import Dataset



def create_landseamask(proj4_target):
    # Define paths and constants
    path_data = '/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/2019/01/'
    proj4_icechart = "+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0"

    crs_ICECHART = CRS.from_proj4(proj4_icechart)
    crs_TARGET = CRS.from_proj4(proj4_target)
    transform_function = Transformer.from_crs(crs_ICECHART, crs_TARGET, always_xy = True)

    # min/max values
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4

    nx = 1845
    ny = 2370

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    nc = Dataset(glob.glob(f"{path_data}ice_conc_svalbard_201901021500.nc")[0], 'r')

    x_input = nc.variables['xc'][:]
    y_input = nc.variables['yc'][:]

    x_diff = x_input[1] - x_input[0]
    y_diff = y_input[1] - y_input[0]

    x_ic = np.pad(x_input, (1,1), 'constant', constant_values=(x_input[0] - x_diff, x_input[-1] + x_diff))
    y_ic = np.pad(y_input, (1,1), 'constant', constant_values=(y_input[0] - y_diff, y_input[-1] + y_diff))

    xx_ic, yy_ic = np.meshgrid(x_ic, y_ic)

    xx_targetgrid, yy_targetgrid = transform_function.transform(xx_ic, yy_ic)
    xx_targetgrid_flat = xx_targetgrid.flatten()
    yy_targetgrid_flat = yy_targetgrid.flatten()

    ice_conc = np.pad(nc.variables['ice_concentration'][...], ((0,0), (1,1), (1,1)), 'constant', constant_values=np.nan)
    ice_conc_flat = ice_conc.flatten()

    SIC_AROMEgrid = griddata((xx_targetgrid_flat, yy_targetgrid_flat), ice_conc_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

    SIC_AROMEgrid = SIC_AROMEgrid[:, :-1]

    nc.close()

    return np.where(np.less(SIC_AROMEgrid, 0.), 1, 0)