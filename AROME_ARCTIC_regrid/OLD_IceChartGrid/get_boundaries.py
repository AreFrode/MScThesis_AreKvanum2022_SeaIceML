# This script is used to get shared boundaries of AROME Arctic and IceCharts after regridding
# Author: Are Frode Kvanum
# Date: 27.02.2022

import numpy as np
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from typing import List, TypeVar

pyproj_crs = TypeVar('pyproj_crs')

def get_common_boundaries(path_current: str, path_target: str, crs_current: pyproj_crs, crs_target: pyproj_crs) -> List[List[List]]:
    """Find common boundaries where current fits inside of target

    kwargs
    path_current: String -- path to current (Lustre)
    path_target: String -- path to target (Lustre)
    crs_current: Pyproj CRS -- crs representation for current's projection
    crs_target: Pyproj CRS -- crs representation for target's projection

    return
    List of values and indexes
    """

    transform_function = Transformer.from_crs(crs_current, crs_target, always_xy=True)

    nc_current = Dataset(path_current, 'r')
    nc_target = Dataset(path_target, 'r')

    x_current = nc_current.variables['x'][:]
    y_current = nc_current.variables['y'][:]
    x_target = nc_target.variables['xc'][:]
    y_target = nc_target.variables['yc'][:]

    xx, yy = np.meshgrid(x_current, y_current)
    # yy, xx = np.meshgrid(y_current, x_current)

    X, Y = transform_function.transform(xx, yy)
    # Y, X = transform_function.transform(yy, xx)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    nearest_min_x = find_nearest(x_target, np.amin(X_flat))
    nearest_max_x = find_nearest(x_target, np.amax(X_flat))
    
    nearest_min_y = find_nearest(y_target, np.amin(Y_flat))
    nearest_max_y = find_nearest(y_target, np.amax(Y_flat))

    nc_current.close()
    nc_target.close()

    return [[nearest_min_x, nearest_max_x], [nearest_min_y, nearest_max_y]]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx].astype(np.int32)

def prettyprint_coords(coords):
    out = f"x lower boundary idx {coords[0][0][0]} with value {coords[0][0][1]}\n\
x upper boundary idx {coords[0][1][0]} with value {coords[0][1][1]}\n\
y lower boundary idx {coords[1][0][0]} with value {coords[1][0][1]}\n\
y upper boundary idx {coords[1][1][0]} with value {coords[1][1][1]}"

    return out
    
def runstuff():
    path_icechart = '/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/2019/01/ice_conc_svalbard_201901021500.nc'
    path_arome = '/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/2019/01/01/arome_arctic_sfx_2_5km_20190101T00Z.nc'
    
    crs_ICECHART = CRS.from_proj4("+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0")
    crs_AROME = CRS.from_proj4('+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06')

    coords = get_common_boundaries(path_arome, path_icechart, crs_AROME, crs_ICECHART)
    
    print(prettyprint_coords(coords))


if __name__ == "__main__":
    runstuff()
