# Script to Regrid Arome Arctic to IceChart projection
# Author: Are Frode Kvanum
# Date: 12.03.2022

import os
import glob
import numpy as np
import itertools
from itertools import chain
from netCDF4 import Dataset
from calendar import monthrange
from pyproj import CRS, Transformer, Proj
from scipy.interpolate import griddata
from datetime import datetime, timedelta

def pad_2d(orig):
	padded = np.zeros(orig.shape[0] + 2)
	padded[1:-1] = np.copy(orig)
	padded[0] = padded[1] - (padded[2] - padded[1])
	padded[-1] = padded[-2] + (padded[2] - padded[1])
	return padded

def runstuff():
	################################################
	# Constants
	################################################
	path_data = '/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/'
	path_output = '/lustre/storeB/users/arefk/AROME_ARCTIC_regrid/'
	path_icechart = '/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/2019/01/ice_conc_svalbard_201901021500.nc'
	proj4_icechart = "+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0"
	proj4_arome = '+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06'
	#
	crs_ICECHART = CRS.from_proj4(proj4_icechart)
	crs_AROME = CRS.from_proj4(proj4_arome)
	transform_function = Transformer.from_crs(crs_AROME, crs_ICECHART, always_xy = True)
	to_lonlat = Proj(proj4_icechart)

	################################################
	# IceChart grid
	################################################
	# min/max values given by get_boundaries.py
	y_min = -2996026
	y_max = -211026
	x_min = -717008
	x_max = 1976992

	#
	# Use the number of points in x,y direction to specify length from the IceChart data
	# Then, based on calculation from get_boandaries.py, only shared indexes are used
	n_y = 2979 - 194
	n_x = 3220 - 526

	x_ICgrid = np.linspace(x_min, x_max, n_x)
	y_ICgrid = np.linspace(y_min, y_max, n_y)
	X_ICgrid, Y_ICgrid = np.meshgrid(x_ICgrid, y_ICgrid)

	################################################
	# Dataset
	################################################
	paths = []
	# for year in range(2019, 2022):
	for year in range(2019, 2020): # Only want one year
		# for month in range(1, 13):
		for month in range(1, 2): # Only want one month
			p = f"{path_data}{year}/{month:02d}/"
			paths.append(p)

	#
	# path_data_task = paths[$SGE_TASK_ID - 1]
	path_data_task = paths[0] # This should be the only path
	print(f"path_data_task = {path_data_task}")
	path_output_task = path_data_task.replace(path_data, path_output)
	print(f"path_output_task = {path_output_task}")
	year_task = path_data_task[len(path_data) : len(path_data) + 4]
	print(f"year_task = {year_task}")
	month_task = path_data_task[len(path_data) + 5 : len(path_data) + 7]
	print(f"month_task = {month_task}")
	nb_days_task = monthrange(int(year_task), int(month_task))[1]
	print(f"nb_days_task = {nb_days_task}")
	#
	if os.path.isdir(path_output_task) == False:
		os.system('mkdir -p ' + path_output_task)

	################################################
	# Data processing
	################################################
	for dd in range(1, nb_days_task + 1):
		yyyymmdd = f"{year_task}{month_task}{dd:02d}"
		print(yyyymmdd)

		T2M_ICgrid = np.zeros((3, n_y, n_x))
		SST_ICgrid = np.zeros((3, n_y, n_x))
		# ZON10M_ICgrid = np.zeros((3, n_y, n_x))

		dataset = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_full_2_5km_{yyyymmdd}T00Z.nc")[0]
	
		nc = Dataset(dataset, 'r')
		# time_arome = nc.variables['time'][:]
		x_input = nc.variables['x'][:]
		y_input = nc.variables['y'][:]
		T2M_input = nc.variables['air_temperature_2m'][:,0,:,:]
		SST_input = nc.variables['air_temperature_0m'][:,0,:,:]
		# ZON10M_input = nc.variables['ZON10M'][:,:,:]

		x_arome = pad_2d(x_input)
		y_arome = pad_2d(y_input)

		xx_arome, yy_arome = np.meshgrid(x_arome, y_arome)

		# T2M_arome = np.full((time_arome.shape[0], y_arome.shape[0], x_arome.shape[0]), np.nan)
		# T2M_arome[:, 1:-1, 1:-1] = np.copy(T2M_input)
		T2M_arome = np.pad(T2M_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
		SST_arome = np.pad(SST_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
		# ZON10M_arome = np.pad(ZON10M_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
    
		xx_ICgrid, yy_ICgrid = transform_function.transform(xx_arome, yy_arome)
		xx_ICgrid_flat = xx_ICgrid.flatten()
		yy_ICgrid_flat = yy_ICgrid.flatten()

		for t in range(0, 3): # 3, as there are almost three full days to mean
			start = 24*t
			stop = start + 24 if t < 2 else start + 18

			T2M_flat = T2M_arome[start:stop,:,:].mean(axis=0).flatten()
			SST_flat = SST_arome[start:stop,:,:].mean(axis=0).flatten()
			# ZON10M_flat = ZON10M_arome[start:stop,:,:].mean(axis=0).flatten()

			T2M_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), T2M_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')
			
			SST_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), SST_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')
			# SST_ICgrid[t] = np.where(SST_ICgrid[t] < 0, np.nan, SST_ICgrid[t]) # Remove invalid temp

			# ZON10M_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), ZON10M_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

		nc.close()
		################################################
		# Output netcdf file
		################################################
		output_filename = 'AROME_T2M_ICgrid_' + yyyymmdd + 'T00Z.nc'
		output_netcdf = Dataset(path_output_task + output_filename, 'w', format = 'NETCDF4')
		#
		y = output_netcdf.createDimension('y', len(y_ICgrid))
		x = output_netcdf.createDimension('x', len(x_ICgrid))
		latitude = output_netcdf.createDimension('latitude', len(y_ICgrid))
		longitude = output_netcdf.createDimension('longitude', len(x_ICgrid))
		time = output_netcdf.createDimension('time', 3)
		#
		yc = output_netcdf.createVariable('yc', 'd', ('y'))
		xc = output_netcdf.createVariable('xc', 'd', ('x'))
		lat = output_netcdf.createVariable('lat', 'd', ('y','x'))
		lon = output_netcdf.createVariable('lon', 'd', ('y','x'))
		timec = output_netcdf.createVariable('time', 'd', ('time'))
		air_temperature = output_netcdf.createVariable('air_temperature', 'd', ('time', 'y', 'x'))
		sea_surface_temp = output_netcdf.createVariable('sea_surface_temperature','d',('time','y','x'))
		# zonal_wind = output_netcdf.createVariable('zonal_wind', 'd', ('time', 'y', 'x'))
		#
		yc.units = 'm'
		yc.standard_name = 'y'
		xc.units = 'm'
		xc.standard_name = 'x'
		timec.units = 'days since the start date'
		timec.standard_name = 'time'
		lat.units='degree'
		lon.units='degree'
		air_temperature.units = 'K'
		air_temperature.standard_name = 'air_temperature'
		sea_surface_temp.units = 'K'
		sea_surface_temp.standard_name = 'sea_surface_temperature'
		# zonal_wind.units = 'm/s'
		# zonal_wind.standard_name = '10m_Zonal_wind'
		#

		yc[:] = np.linspace(y_min, y_max, n_y)
		xc[:] = np.linspace(x_min, x_max, n_x)
		timec[:] = np.linspace(0, 2, 3)
		lons, lats = to_lonlat(X_ICgrid, Y_ICgrid, inverse=True)
		lat[:, :] = lats
		lon[:, :] = lons
		air_temperature[:,:,:] = T2M_ICgrid
		sea_surface_temp[:,:,:] = SST_ICgrid
		# zonal_wind[:,:,:] = ZON10M_ICgrid

		##
		output_netcdf.description = f"{proj4_icechart}"
	
		output_netcdf.close()	
		break

if __name__ == "__main__":
	runstuff()