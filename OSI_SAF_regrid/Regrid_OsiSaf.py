import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from scipy.interpolate import NearestNDInterpolator
from numpy.polynomial import Polynomial
from interpolate import nearest_neighbor_interp

def compute_trend_1d(arr):
    sic_vals = arr[::-1]
    idx = np.isfinite(sic_vals)  # Skip missing entries

    trend = Polynomial.fit(x = range(len(sic_vals[idx])), y = sic_vals[idx], deg = 1)
    return trend.coef[-1]

def is_missing(arr):
    nans = np.isnan(arr)
    return nans.all()

def has_missing_data(status_flags, xx_arome_flat, yy_arome_flat, x_target, y_target):
    status_flags_flat = (np.pad(status_flags, ((1,1), (1,1)), 'constant', constant_values=np.nan)).flatten()
    status_flags_arome = griddata((xx_arome_flat, yy_arome_flat), status_flags_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')
    return (status_flags_arome == 101).any()

def mask_land(ice_conc, fill_value):
    mask = np.where(~(ice_conc==fill_value))
    mask_T = np.transpose(mask)

    sic_interpolator = NearestNDInterpolator(mask_T, ice_conc[mask])
    return sic_interpolator(*np.indices(ice_conc.shape))

def main():
    # Define paths
    PATH_OSISAF = '/lustre/storeB/project/copernicus/osisaf/data/archive/ice/conc/'
    PATH_OUTPUT = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/Data/'

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"
    proj4_osisaf = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45"

    crs_AROME = CRS.from_proj4(proj4_arome)
    crs_OSISAF = CRS.from_proj4(proj4_osisaf)
    transform_function = Transformer.from_crs(crs_OSISAF, crs_AROME, always_xy = True)

    # How does "real-time" conditions compare against "climatological" conditions?
    num_days = [3, 5, 7, 9, 11, 31]
    n_trends = len(num_days)

    # AROME grid
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4
    
    nx = 1845
    ny = 2370

    xc_osisaf = 760
    yc_osisaf = 1120

    x_target = np.linspace(x_min, x_max, nx)
    y_target = np.linspace(y_min, y_max, ny)

    # Dataset
    paths = []
    for year in range(2016, 2023):
        for month in range(1, 13):
            p = f"{PATH_OSISAF}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[int(sys.argv[1]) - 1] # This should be the only path
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
    baltic_mask = np.zeros((yc_osisaf, xc_osisaf))
    baltic_mask[610:780, 625:700] = 1

    # Data processing
    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        
        path_osisaf = glob.glob(f"{path_data_task}/ice_conc_nh_polstere-100_multi_{yyyymmdd}1200.nc")

        # Read input
        raw_ice_conc = np.empty((num_days[-1], yc_osisaf, xc_osisaf))
        raw_ice_conc.fill(np.nan)

        try:
            dataset = path_osisaf[0]

        except IndexError:
            continue

        with Dataset(dataset, 'r') as nc:
            x_input = nc.variables['xc'][:] * 1000
            y_input = nc.variables['yc'][:] * 1000
            lsmask = np.where(nc.variables['status_flag'][:] == 100, 1, 0)

            xx, yy = np.meshgrid(x_input, y_input)

            xx_regrid, yy_regrid = transform_function.transform(xx, yy)

            lat_input = np.expand_dims(nc.variables['lat'][:], axis = 0)
            lon_input = np.expand_dims(nc.variables['lon'][:], axis = 0)

            tmp_ice_conc = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = -999)
            tmp_ice_conc = np.where(baltic_mask == 1, -999, tmp_ice_conc)

            raw_ice_conc[0] = mask_land(tmp_ice_conc, -999)


        for i in range(1, num_days[-1]):
            yyyymmdd_current = (yyyymmdd_datetime - timedelta(days = i)).strftime('%Y%m%d')

            try:
                path_current = glob.glob(f"{PATH_OSISAF}{yyyymmdd_current[:4]}/{yyyymmdd_current[4:6]}/ice_conc_nh_polstere-100_multi_{yyyymmdd_current}1200.nc")[0]
            
            # If missing days, compute trend from remainder of days
            except IndexError:
                continue

            with Dataset(path_current, 'r') as nc:
                tmp_ice_conc = np.ma.filled(nc.variables['ice_conc'][0,:,:], fill_value = -999)
                tmp_ice_conc = np.where(baltic_mask == 1, -999, tmp_ice_conc)

                raw_ice_conc[i] = mask_land(tmp_ice_conc, -999)

        # Interpolate osisaf as with raw icecharts, NN over land
        
        trend_array = np.zeros((n_trends, yc_osisaf, xc_osisaf))

        for i in range(len(num_days)):
            valid_length = [~np.isnan(raw_ice_conc[j, :, :]).all() for j in range(num_days[i])]
            valid_days = np.sum(valid_length)

            if valid_days > 1:
                trend_array[i] = np.apply_along_axis(compute_trend_1d, axis = 0, arr  = raw_ice_conc[:num_days[i],:,:])

            else:
                trend_array[i] = -999

        target_array = np.concatenate((lat_input, lon_input, lsmask, trend_array), axis = 0)

        interp_array = nearest_neighbor_interp(xx_regrid, yy_regrid, x_target, y_target, target_array)
        
        # Write to file
        output_filename = f"OSISAF_trend_1kmgrid_{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", "w", format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(x_target))
            nc_out.createDimension('y', len(y_target))
            nc_out.createDimension('trend', len(num_days))

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = y_target
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = x_target

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees_north'
            latc.standard_name = 'Latitude'
            latc[:] = interp_array[0]

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees_east'
            lonc.standard_name = 'Longitude'
            lonc[:] = interp_array[1]

            lsmaskc = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmaskc.units = 'binary'
            lsmaskc.standard_name = 'Land Sea mask'
            lsmaskc[:] = interp_array[2]

            ice_conc_trend_out = nc_out.createVariable('ice_conc_trend', 'd', ('trend', 'y', 'x'))
            ice_conc_trend_out.units = '%'
            ice_conc_trend_out.standard_name = 'Sea Ice Concentration Trend'
            ice_conc_trend_out[:] = interp_array[3:]

if __name__ == "__main__":
    main()
