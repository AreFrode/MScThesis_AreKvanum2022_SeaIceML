import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from datetime import datetime, timedelta
from pyproj import CRS, Transformer

from Regrid_OsiSaf import compute_trend_1d
from common_functions import onehot_encode_sic_numerical, get_ml_domain_borders

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    path_amsr2 = f"/lustre/storeB/users/maltem/AIS/seaice/amsr2/"
    # alt_path_amsr2 = f"/lustre/storeB/users/arefk/amsr2/"
    path_amsr2_commons = f"/lustre/storeB/users/maltem/AIS/seaice/amsr2/r.AMSR2-2022.nc"
    path_amsr2_latlon = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/LongitudeLatitudeGrid-n6250-Arctic.nc"

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/amsr2-trend/"

    path_ml = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/weights_21021550/2022/01/"

    # Define projection transformer
    epsg_amsr2 = 3411
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_AMSR2 = CRS.from_epsg(epsg_amsr2)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_AMSR2, always_xy = True)

    xxc_target_flat, yyc_target_flat = get_ml_domain_borders(path_ml, transform_function)

    num_days = 7
    
    # Define months for parallel execution
    year = 2022
    months = []
    for month in range(1, 13):
        months.append(month)

    month_task = months[int(sys.argv[1]) - 1]
    print(f"{month_task=}")

    path_output_task = f"{path_output}{year}/{month_task:02d}/"
    print(f"{path_output_task=}")

    nb_days_task = monthrange(int(year), int(month_task))[1]
    print(f"{nb_days_task=}")
    #
    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    with Dataset(path_amsr2_commons, 'r') as nc_commons:
        amsr2_x = nc_commons.variables['x'][:]
        amsr2_y = nc_commons.variables['y'][:]

    leftmost_boundary = find_nearest(amsr2_x, np.min(xxc_target_flat))
    rightmost_boundary = find_nearest(amsr2_x, np.max(xxc_target_flat)) + 1

    lower_boundary = find_nearest(amsr2_y, np.min(yyc_target_flat))
    upper_boundary = find_nearest(amsr2_y, np.max(yyc_target_flat)) + 1

    amsr2_x_current = amsr2_x[leftmost_boundary:rightmost_boundary]
    amsr2_y_current = amsr2_y[lower_boundary:upper_boundary]

    # Define baltic mask
    baltic_mask = np.zeros((len(amsr2_y_current), len(amsr2_x_current)))
    baltic_mask[:210, 240:] = 1

    sic_key = 'z'

    with Dataset(path_amsr2_latlon, 'r') as constants:
        amsr2_lat = constants['Latitudes'][:]
        amsr2_lon = (constants['Longitudes'][:] + 180) % 360 - 180

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year}{month_task:02d}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')

        read_fields = []

        try:
            amsr2_path = glob.glob(f"{path_amsr2}{year}/asi-AMSR2-n6250-{yyyymmdd}-v5.4.nc")[0]

        except IndexError:
            continue


        with Dataset(amsr2_path, 'r') as nc:
            amsr2_sic = nc.variables[sic_key][:]
            
        amsr2_sic_current = np.ma.filled(amsr2_sic[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary], fill_value = 0)

        amsr2_lsmask = np.isnan(amsr2_sic_current).astype(int)

        amsr2_sic_current = np.where(baltic_mask == 1, 0, amsr2_sic_current)
        amsr2_sic_current = np.where(amsr2_lsmask == 1, np.nan, amsr2_sic_current)

        read_fields.append(amsr2_sic_current)
        

        for i in range(1, num_days):
            yyyymmdd_current = (yyyymmdd_datetime - timedelta(days = i)).strftime('%Y%m%d')

            try:
                path_current = glob.glob(f"{path_amsr2}{yyyymmdd_current[:4]}/asi-AMSR2-n6250-{yyyymmdd_current}-v5.4.nc")[0]
            
            # If missing days, compute trend from remainder of days
            except IndexError:
                continue
                
            with Dataset(path_current, 'r') as nc:
                trend_sic = np.ma.filled(nc.variables[sic_key][lower_boundary:upper_boundary,leftmost_boundary:rightmost_boundary], fill_value = 0)

                trend_sic = np.where(baltic_mask == 1, 0, trend_sic)
                trend_sic = np.where(amsr2_lsmask == 1, np.nan, trend_sic)

                read_fields.append(trend_sic)

        read_fields = np.array(read_fields)

        if len(read_fields) == 7:
            trend_array = np.apply_along_axis(compute_trend_1d, axis = 0, arr  = read_fields)

        else:
            continue
        
        ny, nx = read_fields[0].shape

        ice_conc_days = np.zeros((3, ny, nx))

        for j in range(3):
            ice_conc_days[j] = read_fields[0] + (j + 1) * trend_array
        
        ice_conc_days[ice_conc_days < 0] = 0
        ice_conc_days[ice_conc_days > 100] = 100

        output_filename = f"amsr2_trend_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', nx)
            nc_out.createDimension('y', ny)
            nc_out.createDimension('t', 3)

            sic_orig = nc_out.createVariable('sic_orig', 'd', ('y', 'x'))
            sic_orig.units = "1"
            sic_orig.standard_name = "Origin Sea Ice Concentration"
            sic_orig[:] = onehot_encode_sic_numerical(read_fields[0])

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(ice_conc_days)


if __name__ == "__main__":
    main()