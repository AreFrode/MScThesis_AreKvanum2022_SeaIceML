#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-36
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/PROG/regrid_arome_1km_""$SGE_TASK_ID"".py" << EOF

#######################################################################################################
import glob
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from scipy.interpolate import griddata
from rotate_wind_from_UV_to_xy import rotate_wind_from_UV_to_xy


def main():
    ################################################
	# Constants
	################################################
    path_data = '/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/'
    path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/'

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    ################################################
	# IceChart (AromeArctic 1km x 1km) grid
	################################################
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4

    nx = 1845
    ny = 2370

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    ################################################
    # Dataset
    ################################################
    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
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

        path_day = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_full_2_5km_{yyyymmdd}T00Z.nc")
        path_day_sfx = glob.glob(f"{path_data_task}{dd:02d}/arome_arctic_sfx_2_5km_{yyyymmdd}T00Z.nc")

        try:
            dataset = path_day[0]
            dataset_sfx = path_day_sfx[0]

        except IndexError:
            continue

        # Fetch variables from Arome Arctic
        nc = Dataset(dataset, 'r')
        nc_sfx = Dataset(dataset_sfx, 'r')

        x_input = nc.variables['x'][:]
        y_input = nc.variables['y'][:]

        xx_input, yy_input = np.meshgrid(x_input, y_input)
        xx_input_flat = xx_input.flatten()
        yy_input_flat = yy_input.flatten()
        
        lat_arome = nc.variables['latitude'][:]
        lon_arome = nc.variables['longitude'][:]
        t2m_arome = nc.variables['air_temperature_2m'][:]
        uwind_arome = nc.variables['x_wind_10m'][:]
        vwind_arome = nc.variables['y_wind_10m'][:]
        sst_arome = nc_sfx.variables['SST'][:]

        # Allocate target arrays
        lat_target = np.zeros((ny, nx))
        lon_target = np.zeros((ny, nx))
        sst_target = np.zeros((ny, nx))
        t2m_target = np.zeros((3, ny, nx))
        xwind_target = np.zeros((3, ny, nx))
        ywind_target = np.zeros((3, ny, nx))


        # Regrid and assign
        lat_target[...] = griddata((xx_input_flat, yy_input_flat), lat_arome.flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')
        lon_target[...] = griddata((xx_input_flat, yy_input_flat), lon_arome.flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')
        sst_target[...] = griddata((xx_input_flat, yy_input_flat), sst_arome[0, ...].flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')
        
        for t in range(3):
            start = t*24
            stop = start + 24 if t != 2 else None
            t2m_flat = t2m_arome[start:stop, ...].mean(axis=0).flatten()
            uwind_flat = uwind_arome[start:stop, ...].mean(axis=0).flatten()
            vwind_flat = vwind_arome[start:stop, ...].mean(axis=0).flatten()

            t2m_target[t] = griddata((xx_input_flat, yy_input_flat), t2m_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')
            uwind_target = griddata((xx_input_flat, yy_input_flat), uwind_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')
            vwind_target = griddata((xx_input_flat, yy_input_flat), vwind_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

            xwind_target[t], ywind_target[t] = rotate_wind_from_UV_to_xy(x_target, y_target, proj4_arome, uwind_target, vwind_target)


        nc.close()
        nc_sfx.close()

        ################################################
        # Output netcdf file
        ################################################
        output_filename = f"AROME_1kmgrid_{yyyymmdd}T00Z.nc"
        output_netcdf = Dataset(f"{path_output_task}{output_filename}", 'w', format = 'NETCDF4')

        output_netcdf.createDimension('y', len(y_target))
        output_netcdf.createDimension('x', len(x_target))
        output_netcdf.createDimension('t', 3)

        yc = output_netcdf.createVariable('y', 'd', ('y'))
        yc.units = 'm'
        yc.standard_name = 'y'

        xc = output_netcdf.createVariable('x', 'd', ('x'))
        xc.units = 'm'
        xc.standard_name = 'x'

        tc = output_netcdf.createVariable('t', 'd', ('t'))
        tc.units = 'days since forecast inception'
        tc.standard_name = 'time'

        latc = output_netcdf.createVariable('lat', 'd', ('y', 'x'))
        latc.units = 'degrees_north'
        latc.standard_name = 'Latitude'

        lonc = output_netcdf.createVariable('lon', 'd', ('y', 'x'))
        lonc.units = 'degrees_east'
        lonc.standard_name = 'Longitude'

        t2m_out = output_netcdf.createVariable('t2m', 'd', ('t', 'y', 'x'))
        t2m_out.units = 'K'
        t2m_out.standard_name = 'Air temperature at 2 metre height'

        xwind_out = output_netcdf.createVariable('xwind', 'd', ('t', 'y', 'x'))
        xwind_out.units = 'm/s'
        xwind_out.standard_name = 'x 10 metre wind (X10M)'

        ywind_out = output_netcdf.createVariable('ywind', 'd', ('t', 'y', 'x'))
        ywind_out.units = 'm/s'
        ywind_out.standard_name = 'y 10 metre wind (Y10M)'

        sst_out = output_netcdf.createVariable('sst', 'd', ('y', 'x'))
        sst_out.units = 'K'
        sst_out.standard_name = 'Sea Surface Temperature'

        yc[:] = y_target
        xc[:] = x_target
        tc[:] = np.linspace(0,2,3)
        latc[:] = lat_target
        lonc[:] = lon_target
        sst_out[:] = sst_target
        t2m_out[:] = t2m_target
        xwind_out[:] = xwind_target
        ywind_out[:] = ywind_target

        ##
        output_netcdf.description = proj4_arome

        output_netcdf.close()


if __name__ == "__main__":
    main()
########################################################################################################################################################################
EOF

PYTHONPATH=/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/ python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/PROG/regrid_arome_1km_""$SGE_TASK_ID"".py" << EOF