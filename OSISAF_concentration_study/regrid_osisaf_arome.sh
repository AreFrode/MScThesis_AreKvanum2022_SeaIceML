#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-72
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/data_processing_files/PROG/regrid_osisaf_arome_""$SGE_TASK_ID"".py" << EOF

########################################################################################################################################################################
import glob
import os

import numpy as np

from netCDF4 import Dataset
from calendar import monthrange

from pyproj import CRS, Transformer
from scipy.interpolate import griddata

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def get_access_range(path_data, start, stop):
    paths = []
    for year in range(start, stop + 1):
        for month in range(1, 13):
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)

    return paths

def get_monthrange(path_data, path_output, paths):
    path_output_task = paths.replace(path_data, path_output)
    year_task = paths[len(path_data) : len(path_data) + 4]
    month_task = paths[len(path_data) + 5 : len(path_data) + 7]
    nb_days_task = monthrange(int(year_task), int(month_task))[1]

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)
	
    return year_task, month_task, nb_days_task, path_output_task

def main_loop(path_osisaf, transform_function, x_arometarget, y_arometarget, ice_conc_out, dd):
    nc = Dataset(path_osisaf)

    x_osisaf = nc.variables['xc'][:] * 1000
    y_osisaf = nc.variables['yc'][:] * 1000
    # lat_osisaf = nc.variables['lat'][:]
    # lon_osisaf = nc.variables['lon'][:]
    ice_conc = nc.variables['ice_conc'][0,...]

    xx_osisaf, yy_osisaf = np.meshgrid(x_osisaf, y_osisaf)
    xx_aromegrid, yy_aromegrid = transform_function.transform(xx_osisaf, yy_osisaf)

    xx_aromegrid_flat = xx_aromegrid.flatten()
    yy_aromegrid_flat = yy_aromegrid.flatten()
    # lat_flat = lat_osisaf.flatten()
    # lon_flat = lon_osisaf.flatten()
    ice_conc_flat = ice_conc.flatten()

    # Regrid across entire domain

    # lat_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), lat_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')
    # lon_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), lon_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')
    ice_conc_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), ice_conc_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')

    ice_conc_out[dd-1] = ice_conc_aromegrid

    nc.close()

def runstuff():
    # Define paths
    osisaf_pre2015 = "/lustre/storeB/project/copernicus/osisaf/data/reprocessed/ice/conc/v2p0/"
    osisaf_post2015 = "/lustre/storeB/project/copernicus/osisaf/data/reprocessed/ice/conc-cont-reproc/v2p0/"
    path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/Data/'


    arome_proj4 = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"
    osisaf_proj4 = "+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0"
    
    CRS_arome = CRS.from_proj4(arome_proj4)
    CRS_osisaf = CRS.from_proj4(osisaf_proj4)

    transform_function = Transformer.from_crs(CRS_osisaf, CRS_arome, always_xy=True)

    x_arometarget = np.linspace(278603.2, 2123603, 739)
    y_arometarget = np.linspace(-897931.6, 1472068, 949)

    # paths = get_access_range(osisaf_pre2015, 2012, 2016)  # uncomment if pre 2016
    paths = get_access_range(osisaf_post2015, 2016, 2022) # uncomment if post 2015

    year, month, days, output = get_monthrange(osisaf_post2015, path_output, paths[$SGE_TASK_ID - 1])

    ice_conc_out = np.zeros((days, 949, 739))
    
    for dd in range(1, days):
        yyyymmdd = f"{year}{month}{dd:02d}"
        print(yyyymmdd)

        try:
            # path_osisaf = glob.glob(f"{paths[$SGE_TASK_ID - 1]}ice_conc_nh_ease2-250_cdr-v2p0_{yyyymmdd}1200.nc")[0]  # uncomment if pre 2016
            path_osisaf = glob.glob(f"{paths[$SGE_TASK_ID - 1]}ice_conc_nh_ease2-250_icdr-v2p0_{yyyymmdd}1200.nc")[0]  # uncomment if post 2015
        
        except IndexError:
            continue

        main_loop(path_osisaf, transform_function, x_arometarget, y_arometarget, ice_conc_out, dd)
        

    dd = days
    yyyymmdd = f"{year}{month}{dd:02d}"
    print(yyyymmdd)

    try:
        # path_osisaf = glob.glob(f"{paths[$SGE_TASK_ID - 1]}ice_conc_nh_ease2-250_cdr-v2p0_{yyyymmdd}1200.nc")[0]  # uncomment if pre 2016
        path_osisaf = glob.glob(f"{paths[$SGE_TASK_ID - 1]}ice_conc_nh_ease2-250_icdr-v2p0_{yyyymmdd}1200.nc")[0]  # uncomment if post 2015
    
    except IndexError:
        print('Final day missing, error')
        exit()

    nc = Dataset(path_osisaf)

    x_osisaf = nc.variables['xc'][:] * 1000
    y_osisaf = nc.variables['yc'][:] * 1000
    lat_osisaf = nc.variables['lat'][:]
    lon_osisaf = nc.variables['lon'][:]
    ice_conc = nc.variables['ice_conc'][0,...]

    xx_osisaf, yy_osisaf = np.meshgrid(x_osisaf, y_osisaf)
    xx_aromegrid, yy_aromegrid = transform_function.transform(xx_osisaf, yy_osisaf)

    xx_aromegrid_flat = xx_aromegrid.flatten()
    yy_aromegrid_flat = yy_aromegrid.flatten()
    lat_flat = lat_osisaf.flatten()
    lon_flat = lon_osisaf.flatten()
    ice_conc_flat = ice_conc.flatten()

    # Regrid across entire domain

    lat_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), lat_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')
    lon_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), lon_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')
    ice_conc_aromegrid = griddata((xx_aromegrid_flat, yy_aromegrid_flat), ice_conc_flat, (x_arometarget[None,:], y_arometarget[:,None]), method='nearest')

    ice_conc_out[dd-1] = ice_conc_aromegrid
    ice_conc_out = np.ma.masked_where(ice_conc_out < 0., ice_conc_out)

    nc.close()

    # Output netcdf
    output_filename = f"OSISAF_AROMEgrid_{year}{month}T1200.nc"
    output_netcdf = Dataset(f"{output}{output_filename}", "w", format = "NETCDF4")

    x = output_netcdf.createDimension('x', len(x_arometarget))
    y = output_netcdf.createDimension('y', len(y_arometarget))
    time = output_netcdf.createDimension('t', days)

    xc = output_netcdf.createVariable('xc', 'd', ('x'))
    xc.units = 'm'

    yc = output_netcdf.createVariable('yc', 'd', ('y'))
    yc.units = 'm'

    latc = output_netcdf.createVariable("lat", 'd', ('y', 'x'))
    latc.units = 'degrees_north'

    lonc = output_netcdf.createVariable("lon", 'd', ('y', 'x'))
    lonc.units = 'degrees_east'
    
    ice_concentration = output_netcdf.createVariable("ice_conc", 'd', ('t', 'y', 'x'))
    ice_concentration.units = 'Sea Ice Concentration (%)'

    xc[:] = x_arometarget
    yc[:] = y_arometarget
    latc[:] = lat_aromegrid
    lonc[:] = lon_aromegrid
    ice_concentration[...] = ice_conc_out


    output_netcdf.description = arome_proj4
    output_netcdf.close()


if __name__ == "__main__":
    runstuff()

########################################################################################################################################################################
EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/data_processing_files/PROG/regrid_osisaf_arome_""$SGE_TASK_ID"".py"