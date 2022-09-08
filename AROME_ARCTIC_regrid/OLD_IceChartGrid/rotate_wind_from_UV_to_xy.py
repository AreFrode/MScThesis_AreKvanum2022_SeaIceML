import numpy as np
from pyproj import CRS, Transformer
from netCDF4 import Dataset

#
def rotate_wind_from_UV_to_xy(x, y, grid_proj4_str, U, V):  # x and y must be vectors and in meters. 
	###############
	# Declaration of the variables
	###############
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	x_extend = np.full(len(x) + 1, np.nan)
	x_extend[0:-1] = x
	x_extend[-1] = x[-1] + dx
	y_extend = np.full(len(y) + 1, np.nan) 
	y_extend[0:-1] = y
	y_extend[-1] = y[-1] + dy
	#
	xx, yy = np.meshgrid(x_extend, y_extend)
	crs_grid = CRS.from_proj4(grid_proj4_str)
	crs_EPSG4326 = CRS.from_proj4('+proj=latlon')
	transform_function = Transformer.from_crs(crs_grid, crs_EPSG4326, always_xy = True)
	lon, lat = transform_function.transform(xx, yy)
	#
	idx_y = np.arange(len(y))
	idx_x = np.arange(len(x))
	lat_y1 = np.full(np.shape(lat), np.nan)
	lat_x1 = np.full(np.shape(lat), np.nan)
	lon_y1 = np.full(np.shape(lat), np.nan)
	lon_x1 = np.full(np.shape(lat), np.nan)
	sign_x = np.full((len(y), len(x)), np.nan)
	sign_y = np.full((len(y), len(x)), np.nan)
	wind_x = np.full((len(y), len(x)), np.nan)
	wind_y = np.full((len(y), len(x)), np.nan)
	###############
	# Wind speed and direction
	###############
	WS = np.sqrt(U ** 2 + V ** 2)
	WD = (180 / np.pi) * np.arctan2(U , V) % 360  # Wind direction (opposite to the meteorological wind direction, direction the wind is blowing to)
	###############
	# Calculate bearing angles of the axes
	###############
	lat_x1[:, 0:-1] = lat[:, idx_x + 1]
	lon_x1[:, 0:-1] = lon[:, idx_x + 1]
	lat_y1[0:-1, :] = lat[idx_y + 1, :]
	lon_y1[0:-1, :] = lon[idx_y + 1, :]
	#
	lat_rad = np.radians(lat)
	lat_x1_rad = np.radians(lat_x1)
	lat_y1_rad = np.radians(lat_y1)
	#
	diff_lon_x = np.radians(lon_x1 - lon)
	xbear_x = np.sin(diff_lon_x) * np.cos(lat_x1_rad)
	ybear_x = np.cos(lat_rad) * np.sin(lat_x1_rad) - (np.sin(lat_rad) * np.cos(lat_x1_rad) * np.cos(diff_lon_x))
	initial_bearing_x = np.degrees(np.arctan2(xbear_x, ybear_x))
	compass_bearing_x_axis = (initial_bearing_x + 360) % 360
	compass_bearing_x_axis = compass_bearing_x_axis[0:-1, 0:-1]
	#
	diff_lon_y = np.radians(lon_y1 - lon)
	xbear_y = np.sin(diff_lon_y) * np.cos(lat_y1_rad)
	ybear_y = np.cos(lat_rad) * np.sin(lat_y1_rad) - (np.sin(lat_rad) * np.cos(lat_y1_rad) * np.cos(diff_lon_y))
	initial_bearing_y = np.degrees(np.arctan2(xbear_y, ybear_y))
	compass_bearing_y_axis = (initial_bearing_y + 360) % 360
	compass_bearing_y_axis = compass_bearing_y_axis[0:-1, 0:-1]
	###############
	# Sign of the wind components along the x and y axes
	###############
	min_positive_x_angle = (compass_bearing_x_axis - 90) % 360
	max_positive_x_angle = (compass_bearing_x_axis + 90) % 360
	min_positive_y_angle = (compass_bearing_y_axis - 90) % 360
	max_positive_y_angle = (compass_bearing_y_axis + 90) % 360
	#
	sign_x[np.logical_and(min_positive_x_angle < 180, np.logical_and(WD > min_positive_x_angle, WD < max_positive_x_angle) == True)] = 1
	sign_x[np.logical_and(min_positive_x_angle < 180, np.logical_and(WD > min_positive_x_angle, WD < max_positive_x_angle) == False)]  = -1
	sign_x[np.logical_and(min_positive_x_angle >= 180, np.logical_or(WD > min_positive_x_angle, WD < max_positive_x_angle) == True)] = 1
	sign_x[np.logical_and(min_positive_x_angle >= 180, np.logical_or(WD > min_positive_x_angle, WD < max_positive_x_angle) == False)] = -1
	sign_y[np.logical_and(min_positive_y_angle < 180, np.logical_and(WD > min_positive_y_angle, WD < max_positive_y_angle) == True)] = 1
	sign_y[np.logical_and(min_positive_y_angle < 180, np.logical_and(WD > min_positive_y_angle, WD < max_positive_y_angle) == False)]  = -1
	sign_y[np.logical_and(min_positive_y_angle >= 180, np.logical_or(WD > min_positive_y_angle, WD < max_positive_y_angle) == True)] = 1
	sign_y[np.logical_and(min_positive_y_angle >= 180, np.logical_or(WD > min_positive_y_angle, WD < max_positive_y_angle) == False)] = -1
	###############
	# Wind components along the x and y axes
	###############
	Dx = abs(compass_bearing_x_axis - WD)
	Dy = abs(compass_bearing_y_axis - WD)
	#
	wind_x = sign_x * WS * abs(np.cos(Dx * np.pi / 180))
	wind_y = sign_y * WS * abs(np.cos(Dy * np.pi / 180))
	#
	return(wind_x, wind_y)

if __name__ == "__main__":
    path_data = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/2019/01/AROME_ICgrid_20190102T00Z.nc'
    nc = Dataset(path_data, 'r')

    x_ic = nc.variables['xc'][...]
    y_ic = nc.variables['yc'][...]

    proj4_icechart = "+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0"

    U_ic = nc.variables['ZON10M'][...]
    V_ic = nc.variables['MER10M'][...]

    x_wind_ic, y_wind_ic = rotate_wind_from_UV_to_xy(x_ic,y_ic,proj4_icechart,U_ic[0],V_ic[0])
