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

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/PROG/prepare_date_to_hdf_""$SGE_TASK_ID"".py" << EOF

########################################################################################################################################################################
import glob
import h5py
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator

from rotate_wind_from_UV_to_xy import rotate_wind_from_UV_to_xy


def interpolate_invalid(field):
    """NN interpolation of NaN areas

    Args:
        field (array): 2d field to apply interpolation

    Returns:
        array: interpolated array
    """
    mask = np.where(~np.isnan(field))
    interp = NearestNDInterpolator(np.transpose(mask), field[mask])
    interp_field = interp(*np.indices(field.shape))
    return np.where(interp_field > 100., -1, interp_field)

def main():
    # setup data-paths
    path_Arome = "/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/"
    path_RegridIceChart = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ICE_CHART_regrid/Data/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{path_RegridIceChart}{year}/{month:02d}/"
            paths.append(p)

    path_data_task = paths[$SGE_TASK_ID - 1]
    print(path_data_task)
    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_RegridIceChart) : len(path_RegridIceChart) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_RegridIceChart) + 5 : len(path_RegridIceChart) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")

    if not os.path.isdir(f"{path_output}{year_task}/{month_task}"):
        os.makedirs(f"{path_output}{year_task}/{month_task}")

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")

        try:
            ic_path = glob.glob(f"{path_data_task}ICECHART_AROMEgrid_{yyyymmdd}T1500Z.nc")[0]
            arome_path = glob.glob(f"{path_Arome}{year_task}/{month_task}/{dd:02d}/arome_arctic_full_2_5km_{yyyymmdd}T00Z.nc")[0]
            arome_path_sfx = glob.glob(f"{path_Arome}{year_task}/{month_task}/{dd:02d}/arome_arctic_sfx_2_5km_{yyyymmdd}T00Z.nc")[0]


        except IndexError:
            continue

        hdf5_path = f"{path_output}{year_task}/{month_task}/PreparedNNFilled_{yyyymmdd}.hdf5"

        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        outfile = h5py.File(hdf5_path, 'w-')
        
        # Open Ice Chart
        nc = Dataset(ic_path, 'r')
        sic = nc.variables['sic'][:]


        # Open AROME
        nc_a = Dataset(arome_path, 'r')
        nc_a_sfx = Dataset(arome_path_sfx, 'r')

        x = nc_a.variables['x'][:]
        y = nc_a.variables['y'][:]
        lat = nc_a.variables['latitude'][:]
        lon = nc_a.variables['longitude'][:]
        t2m = nc_a.variables['air_temperature_2m'][:]
        uwind = nc_a.variables['x_wind_10m'][:]
        vwind = nc_a.variables['y_wind_10m'][:]
        sst = nc_a_sfx.variables['SST'][0, ...]

        # Write to HDF5
        outfile[f"lat"] = lat[:]
        outfile[f"lon"] = lon[:]
        outfile[f"oobmask"] = np.where(np.isnan(sic), 1, 0)
        outfile[f"sic"] = interpolate_invalid(sic[:])
        outfile[f"sst"] = sst
        outfile[f"lsmask"] = sst.mask.astype(int)

        # Three daily AA means
        for day in range(3):
            key = f"day{day}"
            start = day*24
            stop = start + 24 if day != 2 else None

            outfile[f"{key}/t2m"] = np.mean(t2m[start:stop, 0, :, :], axis=0)

            #Rotate U and V wind to x,y
            xwind, ywind = rotate_wind_from_UV_to_xy(x, y, nc_a.variables['projection_lambert'].proj4, np.mean(uwind[start:stop, 0, :, :], axis=0), np.mean(vwind[start:stop, 0, :, :], axis=0))

            outfile[f"{key}/xwind"] = xwind
            outfile[f"{key}/ywind"] = ywind


        outfile.close()
        nc.close()
        nc_a.close()
        nc_a_sfx.close()



if __name__ == "__main__":
    main()

########################################################################################################################################################################
EOF

PYTHONPATH=/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/ python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/PROG/prepare_date_to_hdf_""$SGE_TASK_ID"".py"