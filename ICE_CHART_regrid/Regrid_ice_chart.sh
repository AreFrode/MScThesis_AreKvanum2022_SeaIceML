#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-36
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/data_processing_files/PROG/regrid_icechart_arome_""$SGE_TASK_ID"".py" << EOF

#######################################################################################################
import glob
import os
import numpy as np
from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata

def runstuff():
	################################################
	# Constants
	################################################
	path_data = '/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/'
	path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/Data/'
	proj4_icechart = "+proj=stere lon_0=0.0 lat_ts=90.0 lat_0=90.0 a=6371000.0 b=6371000.0"
	proj4_arome = '+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06'
	#
	crs_ICECHART = CRS.from_proj4(proj4_icechart)
	crs_AROME = CRS.from_proj4(proj4_arome)
	transform_function = Transformer.from_crs(crs_ICECHART, crs_AROME, always_xy = True)
	# to_lonlat = Proj(proj4_arome)

	################################################
	# Arome Arctic grid
	################################################
	# min/max values
	y_min = -897931.6
	y_max = 1472068
	x_min = 278603.2
	x_max = 2123603

	n_y = 949
	n_x = 739

	x_AROMEgrid = np.linspace(x_min, x_max, n_x)
	y_AROMEgrid = np.linspace(y_min, y_max, n_y)


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
	if not os.path.exists(path_output_task):
		os.makedirs(path_output_task)


	################################################
	# Data processing
	################################################
	for dd in range(1, nb_days_task + 1):
		yyyymmdd = f"{year_task}{month_task}{dd:02d}"
		print(yyyymmdd)

		path_day = glob.glob(f"{path_data_task}ice_conc_svalbard_{yyyymmdd}1500.nc")

		try:
			dataset = path_day[0]

		except IndexError:
			continue

		# Fetch variables from Ice Chart
		nc = Dataset(dataset, 'r')

		# time_input = nc.variables['time'][:]
		x_input = nc.variables['xc'][:]
		y_input = nc.variables['yc'][:]
		x_diff = x_input[1] - x_input[0]
		y_diff = y_input[1] - y_input[0]

		x_ic = np.pad(x_input, (1,1), 'constant', constant_values=(x_input[0] - x_diff, x_input[-1] + x_diff))
		y_ic = np.pad(y_input, (1,1), 'constant', constant_values=(y_input[0] - y_diff, y_input[-1] + y_diff))

		xx_ic, yy_ic = np.meshgrid(x_ic, y_ic)

		xx_AROMEgrid, yy_AROMEgrid = transform_function.transform(xx_ic, yy_ic)
		xx_AROMEgrid_flat = xx_AROMEgrid.flatten()
		yy_AROMEgrid_flat = yy_AROMEgrid.flatten()

		lat_flat = (np.pad(nc.variables['lat'], ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()
		lon_flat = (np.pad(nc.variables['lon'], ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()

		ice_conc = np.pad(nc.variables['ice_concentration'][...], ((0,0), (1,1), (1,1)), 'constant', constant_values=np.nan)
		ice_conc_flat = ice_conc.flatten()

		lat_AROMEgrid = griddata((xx_AROMEgrid_flat, yy_AROMEgrid_flat), lat_flat, (x_AROMEgrid[None, :], y_AROMEgrid[:, None]), method = 'nearest')
		lon_AROMEgrid = griddata((xx_AROMEgrid_flat, yy_AROMEgrid_flat), lon_flat, (x_AROMEgrid[None, :], y_AROMEgrid[:, None]), method = 'nearest')
		SIC_AROMEgrid = griddata((xx_AROMEgrid_flat, yy_AROMEgrid_flat), ice_conc_flat, (x_AROMEgrid[None, :], y_AROMEgrid[:, None]), method = 'nearest')
		SIC_AROMEgrid = np.ma.masked_where(SIC_AROMEgrid < 0., SIC_AROMEgrid)

		nc.close()

		################################################
		# Output netcdf file
		################################################
		output_filename = f"ICECHART_AROMEgrid_{yyyymmdd}T1500Z.nc"
		output_netcdf = Dataset(f"{path_output_task}{output_filename}, 'w', format = 'NETCDF4')

		output_netcdf.createDimension('y', len(y_AROMEgrid))
		output_netcdf.createDimension('x', len(x_AROMEgrid))

		yc = output_netcdf.createVariable('y', 'd', ('y'))
		yc.units = 'm'
		yc.standard_name = 'y'

		xc = output_netcdf.createVariable('x', 'd', ('x'))
		xc.units = 'm'
		xc.standard_name = 'x'

		lat = output_netcdf.createVariable('lat', 'd', ('y', 'x'))
		lat.units = 'degrees_north'
		lat.standard_name = 'Latitude'

		lon = output_netcdf.createVariable('lon', 'd', ('y', 'x'))
		lon.units = 'degrees_east'
		lon.standard_name = 'Longitude'

		sic = output_netcdf.createVariable('sic', 'd', ('y', 'x'))
		sic.units = 'Sea Ice Concentration (%)'
		sic.standard_name = 'Sea Ice Concentration'

		
		# Fill NC variables with values
		yc[:] = y_AROMEgrid[:, None]
		xc[:] = x_AROMEgrid[None, :]
		lat[:] = lat_AROMEgrid
		lon[:] = lon_AROMEgrid 
		sic[:] = SIC_AROMEgrid

		##
		output_netcdf.description = f"{proj4_icechart}"
	
		output_netcdf.close()


if __name__ == "__main__":
	runstuff()
#######################################################################################################
EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/data_processing_files/PROG/regrid_icechart_arome_""$SGE_TASK_ID"".py"