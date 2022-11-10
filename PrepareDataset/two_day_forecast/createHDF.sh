#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-36
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/PROG/prepare_date_two_days_hdf_""$SGE_TASK_ID"".py" << EOF
########################################################################################################################################################################
import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset')

import glob
import os
import h5py

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import NearestNDInterpolator


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
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/two_day_forecast/"
    path_icechart = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    path_osisaf = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/Data/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{path_arome}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[$SGE_TASK_ID - 1]
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
    with Dataset(f"{path_data_task}AROME_1kmgrid_{year_task}{month_task}01T18Z.nc") as constants:
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
        yyyymmdd_datetime = (yyyymmdd_datetime + timedelta(days = 2)).strftime('%Y%m%d')

        try:
            # Assert that arome forecast exist for current day
            # Assert that predictor icechart exist for current day
            # Assert that target icechart exist two timesteps forward in time
            arome_path = glob.glob(f"{path_data_task}AROME_1kmgrid_{yyyymmdd}T18Z.nc")[0]
            icechart_path = glob.glob(f"{path_icechart}{year_task}/{month_task}/ICECHART_1kmAromeGrid_{yyyymmdd}T1500Z.nc")[0]
            target_icechart_path = glob.glob(f"{path_icechart}{yyyymmdd_datetime[:4]}/{yyyymmdd_datetime[4:6]}/ICECHART_1kmAromeGrid_{yyyymmdd_datetime}T1500Z.nc")[0]
            osisaf_path = glob.glob(f"{path_osisaf}{year_task}/{month_task}/OSISAF_trend_1kmgrid_{yyyymmdd}.nc")[0]

        except IndexError:
            continue

        print(arome_path)
        print(icechart_path)
        print(target_icechart_path)
        print(osisaf_path)

        # Prepare output hdf5 file

        hdf5_path = f"{path_output}{year_task}/{month_task}/PreparedSample_{yyyymmdd}.hdf5"

        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        # Open IceChart
        with Dataset(icechart_path, 'r') as nc_ic:
            sic = nc_ic.variables['sic'][:,:-1]
            lat = nc_ic.variables['lat'][:,:-1]
            lon = nc_ic.variables['lon'][:,:-1]
            x = nc_ic.variables['x'][:-1]
            y = nc_ic.variables['y'][:]

        # Open target IceChart
        with Dataset(target_icechart_path, 'r') as nc_ic_target:
            sic_target = nc_ic_target.variables['sic'][:, :-1]

        # Open Arome Arctic
        with Dataset(arome_path, 'r') as nc_a:
            t2m = nc_a.variables['t2m'][:,:,:-1]
            xwind = nc_a.variables['xwind'][:,:,:-1]
            ywind = nc_a.variables['ywind'][:,:,:-1]
            sst = nc_a.variables['sst'][:,:-1]

        with Dataset(osisaf_path, 'r') as nc_osi:
            conc_trend = nc_osi.variables['ice_conc_trend'][0,:,:-1]

        #Apply Wang et.al NearestNeighbor mask to sic
        # sic = np.where(lsmask == 1, np.nan, sic)

        sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
        sic = sic_interpolator(*np.indices(sic.shape))

        sic_target_interpolator = NearestNDInterpolator(mask_T, sic_target[mask])
        sic_target = sic_target_interpolator(*np.indices(sic_target.shape))


        # Write to hdf5
        with h5py.File(hdf5_path, 'w') as outfile:
            outfile['sic'] = onehot_encode_sic(sic)
            outfile['sic_target'] = onehot_encode_sic(sic_target)
            outfile['lon'] = lon
            outfile['lat'] = lat
            outfile['x'] = x
            outfile['y'] = y
            outfile['lsmask'] = lsmask
            outfile['sst'] = sst
            outfile['sic_trend'] = conc_trend

            # Two daily AA means
            for day in range(2):
                key = f"ts{day}"
                outfile[f"{key}/t2m"] = t2m[day,...]
                outfile[f"{key}/xwind"] = xwind[day,...]
                outfile[f"{key}/ywind"] = ywind[day,...]

if __name__ == "__main__":
    main()
########################################################################################################################################################################
EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/PROG/prepare_date_two_days_hdf_""$SGE_TASK_ID"".py"