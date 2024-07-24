import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid")

import pyresample

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator
from pyproj import CRS, Transformer
from tqdm import tqdm

from common_functions import onehot_encode_sic_numerical, find_nearest, get_ml_domain_borders

from fetchFreeDrift import destination_coordinates

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    
    path_amsr2 = f"/lustre/storeB/users/maltem/AIS/seaice/amsr2/"
    alt_path_amsr2 = f"/lustre/storeB/users/arefk/amsr2/"
    path_amsr2_commons = f"/lustre/storeB/users/maltem/AIS/seaice/amsr2/r.AMSR2-2022.nc"
    path_amsr2_latlon = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/LongitudeLatitudeGrid-n6250-Arctic.nc"

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/freedriftAMSR2/"

    path_ml = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/weights_21021550/2022/01/"

    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/arome_winds/"

    # Define projection transformer
    epsg_amsr2 = 3411
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_AMSR2 = CRS.from_epsg(epsg_amsr2)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_AMSR2, always_xy = True)

    xxc_target_flat, yyc_target_flat = get_ml_domain_borders(path_ml, transform_function)

    pi_frac = (180. / np.pi)
    
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

    with Dataset(f"{path_arome}2022/01/arome_winds_b20220101.nc", 'r') as const:
        x = const.variables['x'][:]
        y = const.variables['y'][:]

    arome_area_def = pyresample.geometry.AreaDefinition(
        'ASI-6.25km',
        'PolarStereo',
        'ASI-6.25km',
        epsg_amsr2,
        366,
        368,
        (x[0], y[-1], x[-1], y[0])
    )

    # if month_task >= 11:
        # path_amsr2 = alt_path_amsr2
        # sic_key = 'ASI Ice Concentration'

    with Dataset(path_amsr2_latlon, 'r') as constants:
        amsr2_lat = constants['Latitudes'][:]
        amsr2_lon = (constants['Longitudes'][:] + 180) % 360 - 180

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year}{month_task:02d}{dd:02d}"
        print(yyyymmdd)

        try:
            amsr2_path = glob.glob(f"{path_amsr2}{year}/asi-AMSR2-n6250-{yyyymmdd}-v5.4.nc")[0]
            path_arome_task = glob.glob(f"{path_arome}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/arome_winds_b{yyyymmdd}.nc")[0]

        except IndexError:
            continue

        with Dataset(amsr2_path, 'r') as nc:
            amsr2_sic = nc.variables[sic_key][:]
            
        amsr2_sic_current = np.ma.filled(amsr2_sic[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary], fill_value = np.nan)

        amsr2_lsmask = np.isnan(amsr2_sic_current).astype(int)

        amsr2_sic_current = np.where(baltic_mask == 1, 0, amsr2_sic_current)
        amsr2_sic_current = np.where(amsr2_lsmask == 1, np.nan, amsr2_sic_current)

        output_filename = f"freedriftAMSR2_b{yyyymmdd}.nc"

        with Dataset(path_arome_task, 'r') as nc:
            xwind = nc.variables['xwind'][:]
            ywind = nc.variables['ywind'][:]
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]


        freedrift_sic = -1. * np.ones_like(xwind)
        V_sim = np.empty_like(xwind)

        V_wind = np.sqrt(np.power(xwind, 2) + np.power(ywind, 2))
        Ø_wind = (pi_frac * np.arctan2(xwind, ywind)) % 360

        Ø_sim = (Ø_wind + 20) % 360

        for i in range(3):
            V_sim[i] = 0.02 * (V_wind[i] * (60 * 60 * 24 * (i+1)))


        out_lat, out_lon = destination_coordinates(np.repeat(lat[np.newaxis, ...], 3, axis = 0), np.repeat(lon[np.newaxis, ...], 3, axis = 0), Ø_sim, V_sim)


        for i in range(3):
            t_out_lat = out_lat[i]
            t_out_lon = out_lon[i]

            output = arome_area_def.get_array_indices_from_lonlat(t_out_lon, t_out_lat)

            out_x = output[0]
            out_y = output[1]

            for j in tqdm(range(amsr2_sic_current.shape[0])):
                for k in range(amsr2_sic_current.shape[1]):
                    # print(f"{out_y[j,k]=}")
                    # print(f"{out_x[j,k]=}")
                    if not np.ma.is_masked(out_y[j, k]) and not np.ma.is_masked(out_x[j, k]):
                        if amsr2_sic_current[j,k] > freedrift_sic[i, out_y[j,k], out_x[j,k]]:
                            freedrift_sic[i, out_y[j,k], out_x[j,k]] = amsr2_sic_current[j,k]

            # Neraest Neighbor interpolation to fill missing values
            neg_mask = np.where(~(freedrift_sic[i] == -1.))
            freedrift_sic[i] = NearestNDInterpolator(np.transpose(neg_mask), freedrift_sic[i][neg_mask])(*np.indices(freedrift_sic[i].shape))

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(amsr2_x_current))
            nc_out.createDimension('y', len(amsr2_y_current))
            nc_out.createDimension('t', 3)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = amsr2_y_current
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = amsr2_x_current

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = amsr2_lat[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary]

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Longitude'
            lonc[:] = amsr2_lon[lower_boundary:upper_boundary, leftmost_boundary:rightmost_boundary]

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(freedrift_sic)



if __name__ == "__main__":
    main()