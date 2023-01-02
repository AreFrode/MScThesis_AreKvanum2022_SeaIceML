# The intentions of this script is to seperate the single iecharts.nc provided by Nick into an orderly file structure comprising of folders, subfolders and dates. The file structure will follow the structure of all previous data, as of this point

import glob
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from datetime import datetime, timedelta
from tqdm import tqdm


def main():
    # Define paths
    icechart_path = "/lustre/storeB/users/nicholsh/icecharts_2011-2022.nc"
    previous_icecharts = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/" # match dates
    path_output = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/'


    icecharts = Dataset(icechart_path, 'r')
    icechart_x = icecharts.variables['x'][:]
    icechart_y = icecharts.variables['y'][:]
    icechart_lat = icecharts.variables['lat'][:]
    icechart_lon = icecharts.variables['lon'][:]
    icechart_time = icecharts.variables['time'][:]

    # Define datetime
    t0 = datetime(1981, 1, 1)
        
    for i in tqdm(range(len(icechart_time))):
        time = t0 + timedelta(seconds=int(icechart_time[i]))
        year_task = time.year
        month_task = time.month
        day_task = time.day

        yyyymmdd = f"{year_task}{month_task:02d}{day_task:02}"
        # print(f"{i}: {yyyymmdd}")

        
        path_output_task = f"{path_output}{year_task}/{month_task:02d}/"
        # print(f"path_output_task = {path_output_task}")
        
        if not os.path.exists(path_output_task):
            os.makedirs(path_output_task)

        sic_tmp = icecharts.variables['sic'][i, :]

        outfile = Dataset(f"{path_output_task}ICECHART_1kmAromeGrid_{yyyymmdd}T1500Z.nc", 'w', format = "NETCDF4")

        outfile.createDimension('y', len(icechart_y))
        outfile.createDimension('x', len(icechart_x))

        yc = outfile.createVariable('y', 'd', ('y'))
        yc.units = 'm'
        yc.standard_name = 'y'

        xc = outfile.createVariable('x', 'd', ('x'))
        xc.units = 'm'
        xc.standard_name = 'x'

        lat = outfile.createVariable('lat', 'd', ('y', 'x'))
        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'

        lon = outfile.createVariable('lon', 'd', ('y', 'x'))
        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'

        sic = outfile.createVariable('sic', 'd', ('y', 'x'))
        sic.units = 'sea_ice_concentration (%)'
        sic.standard_name = 'Sea Ice Concentration'

        yc[:] = icechart_y
        xc[:] = icechart_x
        lat[:] = icechart_lat
        lon[:] = icechart_lon
        sic[:] = sic_tmp

        outfile.close()         

    icecharts.close()
    



if __name__ == "__main__":
    main()
