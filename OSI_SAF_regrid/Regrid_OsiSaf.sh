#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-36
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/data_processing_files/PROG/regrid_osisaf_trend_""$SGE_TASK_ID"".py" << EOF

######################################################################################
import glob
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from scipy.interpolate import NearestNDInterpolator
from numpy.polynomial import Polynomial


def compute_trend_1d(arr):
    trend = Polynomial.fit(x = range(len(arr)), y = arr[::-1], deg = 1)
    return trend.coef[-1]


def main():
    # Define paths
    PATH_OSISAF = '/lustre/storeB/project/copernicus/osisaf/data/archive/ice/conc/'
    PATH_OUTPUT = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/Data/'

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"
    proj4_osisaf = "+proj=laea +a=6371228.0 +lat_0=90 +lon_0=0"

    crs_AROME = CRS.from_proj4(proj4_arome)
    crs_OSISAF = CRS.from_proj4(proj4_osisaf)
    transform_function = Transformer.from_crs(crs_OSISAF, crs_AROME, always_xy = True)

    num_days = 5

    # AROME grid
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4
    
    nx = 1845
    ny = 2370

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    # Dataset
    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{PATH_OSISAF}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[$SGE_TASK_ID - 1] # This should be the only path
    print(f"path_data_task = {path_data_task}")
    path_output_task = path_data_task.replace(PATH_OSISAF, PATH_OUTPUT)
    print(f"path_output_task = {path_output_task}")
    year_task = path_data_task[len(PATH_OSISAF) : len(PATH_OSISAF) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(PATH_OSISAF) + 5 : len(PATH_OSISAF) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")
    #
    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)


    # Prepare masks
    baltic_mask = np.zeros((849, 849))
    baltic_mask[575:800, 475:600] = 1

    # Data processing
    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        
        path_osisaf = glob.glob(f"{path_data_task}/ice_conc_nh_ease-125_multi_{yyyymmdd}1200.nc")

        # Read input
        raw_ice_conc = np.empty((849, 849, num_days))

        try:
            dataset = path_osisaf[0]

        except IndexError:
            continue

        with Dataset(dataset, 'r') as nc:
            x_input = nc.variables['xc'][:] * 1000
            y_input = nc.variables['yc'][:] * 1000

            lat_input = nc.variables['lat'][:]
            lon_input = nc.variables['lon'][:]

            tmp_ice_conc = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = -999)
            mask = np.where(~np.logical_or((tmp_ice_conc == -999), (baltic_mask == 1)))
            mask_T = np.transpose(mask)

            sic_interpolator = NearestNDInterpolator(mask_T, tmp_ice_conc[mask])
            raw_ice_conc[..., 0] = sic_interpolator(*np.indices(tmp_ice_conc.shape))
        

        for i in range(1, num_days):
            yyyymmdd_current = (yyyymmdd_datetime - timedelta(days = i)).strftime('%Y%m%d')
            try:
                path_current = glob.glob(f"{PATH_OSISAF}{yyyymmdd_current[:4]}/{yyyymmdd_current[4:6]}/ice_conc_nh_ease-125_multi_{yyyymmdd_current}1200.nc")[0]
            
            # If missing days, compute trend from remainder of days
            except IndexError:
                continue
                
            with Dataset(path_current, 'r') as nc:
                tmp_ice_conc = nc.variables['ice_conc'][0,:,:]
                sic_interpolator = NearestNDInterpolator(mask_T, tmp_ice_conc[mask])
                raw_ice_conc[..., i] = sic_interpolator(*np.indices(tmp_ice_conc.shape))

        x_diff = x_input[1] - x_input[0]
        y_diff = y_input[1] - y_input[0]
        x_ic = np.pad(x_input, (1,1), 'constant', constant_values = (x_input[0] - x_diff, x_input[-1] + x_diff))
        y_ic = np.pad(y_input, (1,1), 'constant', constant_values = (y_input[0] - y_diff, y_input[-1] + y_diff))

        xx_ic, yy_ic = np.meshgrid(x_ic, y_ic)

        xx_arome, yy_arome = transform_function.transform(xx_ic, yy_ic)
        xx_arome_flat = xx_arome.flatten()
        yy_arome_flat = yy_arome.flatten()

        lat_flat = (np.pad(lat_input, ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()
        lon_flat = (np.pad(lon_input, ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()

        raw_ice_conc_flat = (np.pad(raw_ice_conc[..., 0], ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()

        # Interpolate osisaf as with raw icecharts, NN over land
        ice_conc_trend = np.apply_along_axis(compute_trend_1d, axis = -1, arr = raw_ice_conc)

        ice_conc_flat = (np.pad(ice_conc_trend, ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()

        lat_arome = griddata((xx_arome_flat, yy_arome_flat), lat_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

        lon_arome = griddata((xx_arome_flat, yy_arome_flat), lon_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

        ice_conc_arome = griddata((xx_arome_flat, yy_arome_flat), raw_ice_conc_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')

        ice_conc_trend = griddata((xx_arome_flat, yy_arome_flat), ice_conc_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')
        
        ice_conc_days = np.zeros((3, *ice_conc_arome.shape))

        for i in range(3):
            ice_conc_days[i] = ice_conc_arome + (i + 1) * ice_conc_trend

        ice_conc_days[ice_conc_days < 0] = 0
        ice_conc_days[ice_conc_days > 100] = 100

        # Write to file
        output_filename = f"OSISAF_trend_1kmgrid_{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", "w", format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(x_target))
            nc_out.createDimension('y', len(y_target))
            nc_out.createDimension('t', 1)
            nc_out.createDimension('time', 3)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = y_target
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = x_target

            tc = nc_out.createVariable('t', 'd', ('t'))
            tc.units = 'time'
            tc.standard_name = 'time'
            tc[:] = [1]

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees_north'
            latc.standard_name = 'Latitude'
            latc[:] = lat_arome

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees_east'
            lonc.standard_name = 'Longitude'
            lonc[:] = lon_arome

            ice_conc_trend_out = nc_out.createVariable('ice_conc_trend', 'd', ('t', 'y', 'x'))
            ice_conc_trend_out.units = '%'
            ice_conc_trend_out.standard_name = 'Sea Ice Concentration Trend'
            ice_conc_trend_out[:] = ice_conc_trend

            ice_conc_out = nc_out.createVariable('ice_conc', 'd', ('t', 'y', 'x'))
            ice_conc_out.units = '%'
            ice_conc_out.standard_name = 'Sea Ice Concentration'
            ice_conc_out[:] = ice_conc_arome

            ice_conc_days_out = nc_out.createVariable('ice_conc_trend_forecast', 'd', ('time', 'y', 'x'))
            ice_conc_days_out.units = '%'
            ice_conc_days_out.standard_name = 'Sea Ice Concentration trend "time"-days forecast'
            ice_conc_days_out[:] = ice_conc_days


if __name__ == "__main__":
    main()

############################################################################################################

EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/data_processing_files/PROG/regrid_osisaf_trend_""$SGE_TASK_ID"".py" << EOF