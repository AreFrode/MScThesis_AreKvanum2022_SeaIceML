import glob
import os
import h5py

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from landseamask import create_landseamask
from datetime import datetime, timedelta


def onehot_encode_sic(sic):
    fast_ice = np.where(np.equal(sic, 100.), 5, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, 90.), np.less(sic,100.)), 4, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, 70.), np.less(sic, 90.)), 3, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, 40.), np.less(sic, 70.)), 2, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, 10.), np.less(sic, 40.)), 1, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice


def main():
    # setup data-paths
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    path_icechart = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{path_arome}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[0]
    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_arome) : len(path_arome) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_arome) + 5 : len(path_arome) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")

    if not os.path.isdir(f"{path_output}{year_task}/{month_task}"):
        os.makedirs(f"{path_output}{year_task}/{month_task}")


    # Create global land-sea mask
    lsmask = create_landseamask("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06")

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_datetime = (yyyymmdd_datetime + timedelta(days = 7)).strftime('%Y%m%d')

        try:
            arome_path = glob.glob(f"{path_data_task}AROME_1kmgrid_{yyyymmdd}T00Z.nc")[0]
            icechart_path = glob.glob(f"{path_icechart}{year_task}/{month_task}/ICECHART_1kmAromeGrid_{yyyymmdd}T1500Z.nc")[0]
            target_icechart_path = glob.glob(f"{path_icechart}{yyyymmdd_datetime[:4]}/{yyyymmdd_datetime[4:6]}/ICECHART_1kmAromeGrid_{yyyymmdd_datetime}T1500Z.nc")[0]

        except IndexError:
            continue

        # Prepare output hdf5 file

        hdf5_path = f"{path_output}{year_task}/{month_task}/PreparedSample_{yyyymmdd}.hdf5"

        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        outfile = h5py.File(hdf5_path, 'w-')

        # Open IceChart
        nc_ic = Dataset(icechart_path, 'r')
        sic = nc_ic.variables['sic'][:,:-1]
        lat = nc_ic.variables['lat'][:,:-1]
        lon = nc_ic.variables['lon'][:,:-1]

        # Open target IceChart
        nc_ic_target = Dataset(target_icechart_path, 'r')
        sic_target = nc_ic_target.variables['sic'][:, :-1]

        # Open Arome Arctic
        nc_a = Dataset(arome_path, 'r')
        t2m = nc_a.variables['t2m'][:,:,:-1]
        xwind = nc_a.variables['xwind'][:,:,:-1]
        ywind = nc_a.variables['ywind'][:,:,:-1]
        sst = nc_a.variables['sst'][:,:-1]

        # Write to hdf5
        outfile['sic'] = sic
        outfile['sic_target'] = onehot_encode_sic(sic_target)
        outfile['lon'] = lon
        outfile['lat'] = lat
        outfile['lsmask'] = lsmask
        outfile['sst'] = sst

        # Three daily AA means
        for day in range(3):
            key = f"day{day}"
            outfile[f"{key}/t2m"] = t2m[day,...]
            outfile[f"{key}/xwind"] = xwind[day,...]
            outfile[f"{key}/ywind"] = ywind[day,...]


        nc_ic.close()
        nc_ic_target.close()
        nc_a.close()
        outfile.close()

        exit()


if __name__ == "__main__":
    main()