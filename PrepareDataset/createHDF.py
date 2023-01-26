import sys

import glob
import os
import h5py

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import NearestNDInterpolator

from matplotlib import pyplot as plt


def onehot_encode_sic(sic):
    fast_ice = np.where(np.equal(sic, 100.), 6, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, 90.), np.less(sic,100.)), 5, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, 70.), np.less(sic, 90.)), 4, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, 40.), np.less(sic, 70.)), 3, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, 10.), np.less(sic, 40.)), 2, 0)
    open_water = np.where(np.logical_and(np.greater(sic, 0.), np.less(sic, 10.)), 1, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice + open_water


def main():
    # setup data-paths
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    path_icechart = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    path_osisaf = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/Data/"

    # Define lead time in days (1 - 3) and osisaf trend
    lead_times = [1, 2, 3]
    osisaf_trends = [3, 5, 7]

    path_outputs = [f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{i}/" for i in lead_times]

    paths = []
    for year in range(2019, 2023):
        for month in range(1, 13):
            p = f"{path_arome}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[int(sys.argv[1]) - 1]
    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_arome) : len(path_arome) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_arome) + 5 : len(path_arome) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")
    
    for path_output in path_outputs:
        if not os.path.isdir(f"{path_output}{year_task}/{month_task}"):
            os.makedirs(f"{path_output}{year_task}/{month_task}")


    with Dataset(f"{paths[0]}AROME_1kmgrid_20190101T18Z.nc") as constants:
        lsmask = constants['lsmask'][:,:-1]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_targets = [(yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d') for lead_time in lead_times]
        yyyymmdd_osi = (yyyymmdd_datetime + timedelta(days = -1)).strftime('%Y%m%d')

        try:
            # Assert that arome forecast exist for current day
            # Assert that predictor icechart exist for current day
            # Assert that target icechart exist two timesteps forward in time
            arome_path = glob.glob(f"{path_data_task}AROME_1kmgrid_{yyyymmdd}T18Z.nc")[0]
            icechart_path = glob.glob(f"{path_icechart}{year_task}/{month_task}/ICECHART_1kmAromeGrid_{yyyymmdd}T1500Z.nc")[0]
            osisaf_path = glob.glob(f"{path_osisaf}{yyyymmdd_osi[:4]}/{yyyymmdd_osi[4:6]}/OSISAF_trend_1kmgrid_{yyyymmdd_osi}.nc")[0]


        except IndexError:
            continue
        
        # Open IceChart
        with Dataset(icechart_path, 'r') as nc_ic:
            sic = nc_ic.variables['sic'][:,:-1]
            lat = nc_ic.variables['lat'][:,:-1]
            lon = nc_ic.variables['lon'][:,:-1]
            x = nc_ic.variables['x'][:-1]
            y = nc_ic.variables['y'][:]

        #Apply Wang et.al NearestNeighbor mask to sic
        sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
        sic = sic_interpolator(*np.indices(sic.shape))

        # Open OsiSaf trend
        with Dataset(osisaf_path, 'r') as nc_osi:
            conc_trend = nc_osi.variables['ice_conc_trend'][:,:,:-1]

        for i in range(len(lead_times)):
            try:
                target_icechart_path = glob.glob(f"{path_icechart}{yyyymmdd_targets[i][:4]}/{yyyymmdd_targets[i][4:6]}/ICECHART_1kmAromeGrid_{yyyymmdd_targets[i]}T1500Z.nc")[0]

            except IndexError:
                continue

            # Prepare output hdf5 file
            hdf5_path = f"{path_outputs[i]}{year_task}/{month_task}/PreparedSample_v{yyyymmdd_targets[i]}_b{yyyymmdd}.hdf5"

            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

            # Open target IceChart
            with Dataset(target_icechart_path, 'r') as nc_ic_target:
                sic_target = nc_ic_target.variables['sic'][:, :-1]

            sic_target_interpolator = NearestNDInterpolator(mask_T, sic_target[mask])
            sic_target = sic_target_interpolator(*np.indices(sic_target.shape))

            # Open Arome Arctic
            with Dataset(arome_path, 'r') as nc_a:
                t2m = nc_a.variables['t2m'][lead_times[i] - 1,:,:-1]
                xwind = nc_a.variables['xwind'][lead_times[i] - 1,:,:-1]
                ywind = nc_a.variables['ywind'][lead_times[i] - 1,:,:-1]

            # Write to hdf5
            with h5py.File(hdf5_path, 'w') as outfile:
                outfile['sic'] = onehot_encode_sic(sic)
                outfile['sic_target'] = onehot_encode_sic(sic_target)
                outfile['lon'] = lon
                outfile['lat'] = lat
                outfile['x'] = x
                outfile['y'] = y
                outfile['lsmask'] = lsmask
                outfile[f"t2m"] = t2m
                outfile[f"xwind"] = xwind
                outfile[f"ywind"] = ywind

                for i, trend in enumerate(osisaf_trends):
                    outfile[f'osisaf_trend_{trend}/sic_trend'] = conc_trend[i]


if __name__ == "__main__":
    main()