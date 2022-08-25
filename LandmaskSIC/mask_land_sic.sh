#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -t 1-36
#$ -M arefk@met.no
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/data_processing_files/PROG/Bins_""$SGE_TASK_ID""_mask_land_sic.py" << EOF

##################################################################
import glob
import os
import numpy as np

import xarray as xr
from calendar import monthrange
from scipy.ndimage import distance_transform_edt

# This script applies a land mask as outlined in Wang2017
# The land mask is a simple nearest neighbor padding of the SIC over land


def runstuff():
    path_data = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/Data/"

    paths = []
    for year in range(2019, 2022):
    # for year in range(2019, 2020): # Only want one year
        for month in range(1, 13):
        # for month in range(1, 2): # Only want one month
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[$SGE_TASK_ID - 1]
    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_data) : len(path_data) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_data) + 5 : len(path_data) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")
    #

    if not os.path.isdir(f"{path_output}{year_task}/{month_task}"):
        os.makedirs(f"{path_output}{year_task}/{month_task}")

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")

        try:
            ic_path = glob.glob(f"{path_data}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

        except IndexError:
            continue

        output_filename = f"ice_conc_svalbard_landmask_{yyyymmdd}1500.nc"

        icechart = xr.open_dataset(ic_path)

        indices = distance_transform_edt(np.isnan(icechart.ice_concentration[0].values), return_distances=False, return_indices=True)
        icechart['ice_concentration'][0] = icechart.ice_concentration[0].values[tuple(indices)]

        icechart.to_netcdf(f"{path_output}{year_task}/{month_task}/{output_filename}")

if __name__ == "__main__":
    runstuff()

##################################################################
EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/data_processing_files/PROG/Bins_""$SGE_TASK_ID""_mask_land_sic.py"