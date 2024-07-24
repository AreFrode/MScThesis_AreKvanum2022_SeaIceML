import h5py
import glob
import os
import pyresample
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from interpolate import nearest_neighbor_interp
from Regrid_OsiSaf import compute_trend_1d, mask_land
from common_functions import onehot_encode_sic_numerical, get_target_domain
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm


def destination_coordinates(lat1, lon1, initial_bearing, distance): # distance in meters
	RT = 6371.0 # km, but distance must be in meters
	d_div_r = (distance * 0.001) / RT
	lat1 = np.radians(lat1)
	lon1 = np.radians(lon1)
	bearing = np.radians(initial_bearing)
	lat2 = np.arcsin(np.sin(lat1) * np.cos(d_div_r) + np.cos(lat1) * np.sin(d_div_r) * np.cos(bearing))
	lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(d_div_r) * np.cos(lat1), np.cos(d_div_r) - np.sin(lat1) * np.sin(lat2))
	lat2 = np.degrees(lat2)
	lon2 = np.degrees(lon2)

	return(lat2, lon2)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths

    path_icechart = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    common_grid = sys.argv[2]

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_arome, 'freedrift')

    nx = len(target_x)
    ny = len(target_y)

    pi_frac = (180. / np.pi)

    # Define months for parallel execution
    year = 2022
    paths = []
    for month in range(1, 13):
        p = f"{path_icechart}{year}/{month:02d}/"
        paths.append(p)

    path_data_task = paths[int(sys.argv[1]) - 1] # This should be the only path
    print(f"{path_data_task=}")
    path_output_task = path_data_task.replace(path_icechart, path_output)
    print(f"path_output_task = {path_output_task}")
    year_task = path_data_task[len(path_icechart):len(path_icechart) + 4]
    print(f"{year_task=}")
    month_task = path_data_task[len(path_icechart) + 5:len(path_icechart) + 7]
    print(f"{month_task=}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"{nb_days_task=}")
    #
    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    with Dataset(f"{path_arome}2022/01/AROME_1kmgrid_20220101T18Z.nc", 'r') as constants:
        lsmask = constants.variables['lsmask'][:,:-1]
        x = constants.variables['x'][:-1]
        y = constants.variables['y'][:]

    x_diff = x[1] - x[0]
    y_diff = y[1] - y[0]

    xc = np.pad(x, (1,1), 'constant', constant_values = (x[0] - x_diff, x[-1] + x_diff))
    yc = np.pad(y, (1,1), 'constant', constant_values = (y[0] - y_diff, y[-1] + y_diff))

    xxc, yyc = np.meshgrid(xc, yc)
    xxc_target, yyc_target = transform_function.transform(xxc, yyc)

    width = len(x)
    height = len(y)

    arome_area_def = pyresample.geometry.AreaDefinition(
        'AromeArctic-1km',
        'LambertConformalConic',
        'AromeArctic-1km',
        proj4_arome,
        width,
        height,
        (x[0], y[-1], x[-1], y[0])
    )

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)


    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        
        try:
            path_icechart_task = glob.glob(f"{path_data_task}/ICECHART_1kmAromeGrid_{yyyymmdd}T1500Z.nc")[0]
            path_arome_task = glob.glob(f"{path_arome}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/AROME_1kmgrid_{yyyymmdd}T18Z.nc")[0]
        except IndexError:
            continue


        # Read input


        with Dataset(path_icechart_task, 'r') as nc:
            sic = nc.variables['sic'][:,:-1]
            lat = nc.variables['lat'][:,:-1]
            lon = nc.variables['lon'][:,:-1]
            x = nc.variables['x'][:-1]
            y = nc.variables['y'][:]

        sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
        sic = sic_interpolator(*np.indices(sic.shape))

        with Dataset(path_arome_task, 'r') as nc:
            xwind = nc.variables['xwind'][...,:-1]
            ywind = nc.variables['ywind'][...,:-1]

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

            for j in tqdm(range(sic.shape[0])):
                for k in range(sic.shape[1]):
                    # print(f"{out_y[j,k]=}")
                    # print(f"{out_x[j,k]=}")
                    if not np.ma.is_masked(out_y[j, k]) and not np.ma.is_masked(out_x[j, k]):
                        if sic[j,k] > freedrift_sic[i, out_y[j,k], out_x[j,k]]:
                            freedrift_sic[i, out_y[j,k], out_x[j,k]] = sic[j,k]

            # Neraest Neighbor interpolation to fill missing values
            neg_mask = np.where(~(freedrift_sic[i] == -1.))
            freedrift_sic[i] = NearestNDInterpolator(np.transpose(neg_mask), freedrift_sic[i][neg_mask])(*np.indices(freedrift_sic[i].shape))

        


    
        lsmask_padded = np.pad(lsmask, ((1,1), (1,1)), 'constant', constant_values = 1)
        freedrift_sic_padded = np.pad(freedrift_sic, ((0,0), (1,1), (1,1)), 'constant', constant_values = np.nan)

        interp_array = np.concatenate((np.expand_dims(lsmask_padded, axis=0), freedrift_sic_padded))
        
        interp_target = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)

        output_filename = f"freedrift_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(target_x))
            nc_out.createDimension('y', len(target_y))
            nc_out.createDimension('t', 3)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = target_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = target_x

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = target_lat

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Longitude'
            lonc[:] = target_lon

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic_numerical(interp_target[1:])

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interp_target[0]

            

if __name__ == "__main__":
    main()