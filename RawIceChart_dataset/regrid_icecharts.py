import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid')

import os

import numpy as np

from netCDF4 import Dataset
from interpolate import nearest_neighbor_interp
from datetime import datetime, timedelta


def main():
    target_resolution = int(sys.argv[1])

    nick_path = "/lustre/storeB/users/nicholsh/"
    icechart_path = f"{nick_path}icecharts_2011-2022.nc"
    icechart_path_2022 = f"{nick_path}icecharts_2022.nc"
    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/{target_resolution}km/"


    years = []
    for year in range(2019, 2023):
        years.append(year)

    months = []
    for month in range(1, 13):
        months.append(month)

    n_months = len(months)

    thread = int(sys.argv[2])
    year_task = years[int(np.floor((thread - 1) / n_months))]
    month_task = months[(thread - 1) % n_months]

    if year_task < 2022:
        icecharts = Dataset(icechart_path, 'r')
    else:
        icecharts = Dataset(icechart_path_2022, 'r')

    icechart_x = icecharts.variables['x'][:-5]
    icechart_y = icecharts.variables['y'][:]

    xx_icechart, yy_icechart = np.meshgrid(icechart_x, icechart_y)

    icechart_lat = icecharts.variables['lat'][:, :-5]
    icechart_lon = icecharts.variables['lon'][:, :-5]
    icechart_time = icecharts.variables['time'][:]

    # Define target grid
    x_min = icechart_x[0]
    x_max = icechart_x[-1]
    y_min = icechart_y[0]
    y_max = icechart_y[-1]

    nx = int(len(icechart_x) / target_resolution)
    ny = int(len(icechart_y) / target_resolution)

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    # Define datetime
    t0 = datetime(1981, 1, 1)


    baltic_mask = np.zeros_like(icechart_lat)
    baltic_mask[:1200, 1500:] = 1

    for i in range(len(icechart_time)):
        time = t0 + timedelta(seconds=int(icechart_time[i]))
        current_day = time.day

        if year_task != time.year or month_task != time.month:
            continue

        if time.weekday() >= 5:
            continue

        yyyymmdd = f"{year_task}{month_task:02d}{current_day:02}"
        print(f"{yyyymmdd}")

        path_output_task = f"{path_output}{year_task}/{month_task:02d}/"
    
        if not os.path.exists(path_output_task):
            os.makedirs(path_output_task)
    
        icechart_sic = np.where(baltic_mask == 1, 0, icecharts.variables['sic'][i, :, :-5])
        regrid = np.concatenate((icechart_lat[None, :], icechart_lon[None, :], icechart_sic[None, :]))
        if target_resolution > 1:
            regrid = nearest_neighbor_interp(xx_icechart, yy_icechart, x_target, y_target, regrid)


        with Dataset(f"{path_output_task}ICECHART_{target_resolution}kmAromeGrid_{yyyymmdd}T1500Z.nc", "w", format = "NETCDF4") as outfile:
            outfile.createDimension('y', len(y_target))
            outfile.createDimension('x', len(x_target))

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
            sic.units = ('%')
            sic.standard_name = 'Sea Ice Concentration'

            yc[:] = y_target
            xc[:] = x_target
            lat[:] = regrid[0]
            lon[:] = regrid[1]
            sic[:] = regrid[2]
        
    icecharts.close()


if __name__ == "__main__":
    main()
