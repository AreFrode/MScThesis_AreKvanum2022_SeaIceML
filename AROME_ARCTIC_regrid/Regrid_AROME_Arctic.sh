#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-36
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/PROG/AROME_Arctic_IC_regrid_""$SGE_TASK_ID"".py" << EOF

############################################
import glob
from json import load
import netCDF4
import os
import numpy as np
from ast import Slice
from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Proj, Transformer
from scipy.interpolate import griddata
from typing import Tuple
from rotate_wind_from_UV_to_xy import rotate_wind_from_UV_to_xy


def load_3dvariable(id: str, dataset: Dataset, output_dataset: Dataset, name: str, unit: str, standard_name: str, nx: int = 3220 - 526, ny: int = 2979 - 194) -> Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]:
	"""loads a 3d variable, e.g. arome_arctic_full_* ([time, height, x, y])
		Currently very specific for SST


	Args:
		id (str): Name of variable in arome arctic
		dataset (Dataset): dataset where the variable is loaded from
		output_dataset (Dataset): output netcdf file
		name (str): name of variable in output
		unit (str): unit of variable in output
		standard_name (str): descriptive name in output
		nx (int, optional): length of target grid in x direction. Defaults to 3220-526.
		ny (int, optional): length of target grid in y direction. Defaults to 2979-194.

	Returns:
		Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]: output array for storing computed values, arome values and output netCDF4 variable
	"""

	id_ICgrid = np.zeros((3, ny, nx))
	id_input = dataset.variables[id][:,:,:]
	id_input = np.ma.filled(id_input, 0.)
	id_arome = np.pad(id_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
	id_out = output_dataset.createVariable(name, 'd', ('time', 'y', 'x'))
	id_out.units = unit
	id_out.standard_name = standard_name

	return (id_ICgrid, id_arome, id_out)

def load_4dvariable(id: str, dataset: Dataset, output_dataset: Dataset, name: str, unit: str, standard_name: str, nx: int = 3220 - 526, ny: int = 2979 - 194, slicer: Slice = 0) -> Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]:
	"""loads a 4d variable, e.g. arome_arctic_full_* ([time, height, x, y])

	Args:
		id (str): Name of variable in arome arctic
		dataset (Dataset): dataset where the variable is loaded from
		output_dataset (Dataset): output netcdf file
		name (str): name of variable in output
		unit (str): unit of variable in output
		standard_name (str): descriptive name in output
		nx (int, optional): length of target grid in x direction. Defaults to 3220-526.
		ny (int, optional): length of target grid in y direction. Defaults to 2979-194.
		slicer (Slice, optional): height values to obtain. Defaults to 0.

	Returns:
		Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]: output array for storing computed values, arome values and output netCDF4 variable
	"""

	id_ICgrid = np.zeros((3, ny, nx))
	id_input = dataset.variables[id][:,slicer,:,:]
	id_arome = np.pad(id_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
	id_out = output_dataset.createVariable(name, 'd', ('time', 'y', 'x'))
	id_out.units = unit
	id_out.standard_name = standard_name

	return (id_ICgrid, id_arome, id_out)

def load_TCC(id: str, dataset: Dataset, x, y, time, output_dataset: Dataset, name: str, unit: str, standard_name: str, nx: int = 3220 - 526, ny: int = 2979 - 194) -> Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]:
	"""loads a 4d variable, e.g. arome_arctic_full_* ([time, height, x, y])

	Args:
		id (str): Name of variable in arome arctic
		dataset (Dataset): dataset where the variable is loaded from
		output_dataset (Dataset): output netcdf file
		name (str): name of variable in output
		unit (str): unit of variable in output
		standard_name (str): descriptive name in output
		nx (int, optional): length of target grid in x direction. Defaults to 3220-526.
		ny (int, optional): length of target grid in y direction. Defaults to 2979-194.
		slicer (Slice, optional): height values to obtain. Defaults to 0.

	Returns:
		Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]: output array for storing computed values, arome values and output netCDF4 variable
	"""

	id_ICgrid = np.zeros((3, ny, nx))
	id_input = np.zeros((len(time), len(y), len(x)))

	# There has to be a better way than this, slow loop many FLOPs
	for i in range(len(time)):
		id_input[i,:,:] = dataset.variables[id][i,...].sum(axis=0)

	id_arome = np.pad(id_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
	id_out = output_dataset.createVariable(name, 'd', ('time', 'y', 'x'))
	id_out.units = unit
	id_out.standard_name = standard_name

	return (id_ICgrid, id_arome, id_out)


def runstuff():
	################################################
	# Constants
	################################################
	path_data = '/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/'
	path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/'
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
	for year in range(2019, 2022):
	# for year in range(2019, 2020): # Only want one year
		for month in range(1, 13):
		# for month in range(1, 2): # Only want one month
			p = f"{path_data}{year}/{month:02d}/"
			paths.append(p)

	#
	path_data_task = paths[$SGE_TASK_ID - 1]
	# path_data_task = paths[0] # This should be the only path
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

		path_day = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_full_2_5km_{yyyymmdd}T00Z.nc")
		path_day_sfx = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_sfx_2_5km_{yyyymmdd}T00Z.nc")

		if len(path_day) > 0 and len(path_day_sfx) > 0:
			dataset = path_day[0]
			dataset_sfx = path_day_sfx[0]

			# Start setup of NC file at start of loop
			output_filename = 'AROME_ICgrid_' + yyyymmdd + 'T00Z.nc'
			output_netcdf = Dataset(path_output_task + output_filename, 'w', format = 'NETCDF4')

			output_netcdf.createDimension('y', len(y_ICgrid))
			output_netcdf.createDimension('x', len(x_ICgrid))
			output_netcdf.createDimension('latitude', len(y_ICgrid))
			output_netcdf.createDimension('longitude', len(x_ICgrid))
			output_netcdf.createDimension('time', 3)
	
			nc = Dataset(dataset, 'r')
			nc_sfx = Dataset(dataset_sfx, 'r')

			# time_input = nc.variables['time'][:]
			x_input = nc.variables['x'][:]
			y_input = nc.variables['y'][:]
			x_diff = x_input[1] - x_input[0]
			y_diff = y_input[1] - y_input[0]

			x_arome = np.pad(x_input, (1,1), 'constant', constant_values=(x_input[0] - x_diff, x_input[-1] + x_diff))
			y_arome = np.pad(y_input, (1,1), 'constant', constant_values=(y_input[0] - y_diff, y_input[-1] + y_diff))

			xx_arome, yy_arome = np.meshgrid(x_arome, y_arome)

			T2M_ICgrid, T2M_arome, air_temperature = load_4dvariable('air_temperature_2m', nc, output_netcdf, 'T2M', 'K', 'air_temperature')
		
			ZON10M_ICgrid, ZON10M_arome, zonal_wind = load_4dvariable('x_wind_10m', nc, output_netcdf, 'ZON10M', 'm/s', 'Zonal 10 metre wind (U10M)')
		
			MER10M_ICgrid, MER10M_arome, meridional_wind = load_4dvariable('y_wind_10m', nc, output_netcdf, 'MER10M', 'm/s', 'Meridional 10 metre wind (V10M)')

			SST_ICgrid, SST_arome, sea_surf_temp = load_3dvariable('SST', nc_sfx, output_netcdf, 'SST', 'K', 'Sea Surface Temperature')
		
			# TCC_ICgrid, TCC_arome, totalcloudcover = load_TCC('cloud_area_fraction_ml', nc, x_input, y_input, time_input,  output_netcdf, 'TCC', '1', 'Total Cloud Cover / Cloud area fraction')
			HCC_ICgrid, HCC_arome, highcloudcover = load_4dvariable('high_type_cloud_area_fraction', nc, output_netcdf, 'HCC', '1', 'Cloud cover of high clouds')

			MCC_ICgrid, MCC_arome, mediumcloudcover = load_4dvariable('medium_type_cloud_area_fraction', nc, output_netcdf, 'MCC', '1', 'Cloud cover of medium clouds')

			LCC_ICgrid, LCC_arome, lowcloudcover = load_4dvariable('low_type_cloud_area_fraction', nc, output_netcdf, 'LCC', '1', 'Cloud cover of low clouds')

			#
			Xwind_ICgrid = np.zeros_like(T2M_ICgrid)
			Ywind_ICgrid = np.zeros_like(T2M_ICgrid)
			WS_ICgrid = np.zeros_like(T2M_ICgrid)
			WD_ICgrid = np.zeros_like(T2M_ICgrid)
    
			xx_ICgrid, yy_ICgrid = transform_function.transform(xx_arome, yy_arome)
			xx_ICgrid_flat = xx_ICgrid.flatten()
			yy_ICgrid_flat = yy_ICgrid.flatten()

			SST_flat = SST_arome[0, ...].flatten()

			for t in range(0, 3): # 3, as there are almost three full days to mean
				start = 24*t
				stop = start + 24 if t < 2 else start + 18

				# Flatten and daily mean
				T2M_flat = T2M_arome[start:stop,...].mean(axis=0).flatten()
				ZON10M_flat = ZON10M_arome[start:stop,...].mean(axis=0).flatten()
				MER10M_flat = MER10M_arome[start:stop,...].mean(axis=0).flatten()
				HCC_flat = HCC_arome[start:stop,...].mean(axis=0).flatten()
				MCC_flat = MCC_arome[start:stop,...].mean(axis=0).flatten()
				LCC_flat = LCC_arome[start:stop,...].mean(axis=0).flatten()

				T2M_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), T2M_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')
			
				SST_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), SST_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

				ZON10M_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), ZON10M_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')
				MER10M_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), MER10M_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

				HCC_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), HCC_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

				MCC_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), MCC_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

				LCC_ICgrid[t] = griddata((xx_ICgrid_flat, yy_ICgrid_flat), LCC_flat, (x_ICgrid[None, :], y_ICgrid[:, None]), method = 'nearest')

				Xwind_ICgrid[t], Ywind_ICgrid[t] = rotate_wind_from_UV_to_xy(x_ICgrid[None, :].flatten(), y_ICgrid[:,None].flatten(), proj4_icechart, ZON10M_ICgrid[t], MER10M_ICgrid[t])

			WS_ICgrid = np.sqrt(np.power(Xwind_ICgrid, 2) + np.power(Ywind_ICgrid, 2))
			WD_ICgrid = (180 + (180/np.pi)*np.arctan2(Ywind_ICgrid, Xwind_ICgrid)) % 360

			nc.close()
			nc_sfx.close()
			################################################
			# Output netcdf file
			################################################
			#
			yc = output_netcdf.createVariable('yc', 'd', ('y'))
			yc.units = 'm'
			yc.standard_name = 'y'

			xc = output_netcdf.createVariable('xc', 'd', ('x'))
			xc.units = 'm'
			xc.standard_name = 'x'

			lat = output_netcdf.createVariable('lat', 'd', ('y','x'))
			lat.units='degree'
			lon = output_netcdf.createVariable('lon', 'd', ('y','x'))
			lon.units='degree'

			timec = output_netcdf.createVariable('time', 'd', ('time'))
			timec.units = 'days since the start date'
			timec.standard_name = 'time'

			# Additional 3d variables
			xwind = output_netcdf.createVariable("X_wind_10m", 'd', ('time', 'y', 'x'))
			xwind.units = 'm/s'
			xwind.standard_name = "X 10 metre wind (X10M)"

			ywind = output_netcdf.createVariable("Y_wind_10m", 'd', ('time', 'y', 'x'))
			ywind.units = 'm/s'
			ywind.standard_name = "Y 10 metre wind (Y10M)"
		
			wind_speed = output_netcdf.createVariable("10WindSpeed", 'd', ('time', 'y', 'x'))
			wind_speed.units = "m/s"
			wind_speed.standard_name = "10 metre wind speed"

			wind_direction = output_netcdf.createVariable("10WindDirection", 'd', ('time', 'y', 'x'))
			wind_direction.units = "degrees"
			wind_direction.standard_name = "10 metre wind direction"
		

			# Fill NC variables with values
			yc[:] = y_ICgrid[:, None]
			xc[:] = x_ICgrid[None, :]
			timec[:] = np.linspace(0, 2, 3)
			lons, lats = to_lonlat(X_ICgrid, Y_ICgrid, inverse=True)
			lat[:, :] = lats
			lon[:, :] = lons
			air_temperature[...] = T2M_ICgrid
			sea_surf_temp[...] = SST_ICgrid
			zonal_wind[...] = ZON10M_ICgrid
			meridional_wind[...] = MER10M_ICgrid
			xwind[...] = Xwind_ICgrid
			ywind[...] = Ywind_ICgrid
			wind_speed[...] = WS_ICgrid
			wind_direction[...] = WD_ICgrid
			highcloudcover[...] = HCC_ICgrid
			mediumcloudcover[...] = MCC_ICgrid
			lowcloudcover[...] = LCC_ICgrid

			##
			output_netcdf.description = f"{proj4_icechart}"
	
			output_netcdf.close()	

if __name__ == "__main__":
	runstuff()

#################################################
EOF

PYTHONPATH=/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/ python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/PROG/AROME_Arctic_IC_regrid_""$SGE_TASK_ID"".py"