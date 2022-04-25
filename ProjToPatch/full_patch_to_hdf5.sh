#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -M arefk@met.no
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python-devel/3.8.7

cat > "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/data_processing_files/PROG/FULL_Patch_to_hdf5.py" << EOF

###########################################################

import glob
import h5py
import os
import numpy as np
from calendar import monthrange
from netCDF4 import Dataset

# Notes/questions
# have to further anmd generalize this script wo it can work on all files
# Currently have to work on fixing the SIC thresholding

def sliding_window3d(var, x_stride = 100, y_stride = 100):     # 100km x 100km grid
    """Deprecated SlidingWindow, used for testing

    Args:
        var (_type_): _description_
        x_stride (int, optional): _description_. Defaults to 100.
        y_stride (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    len_y, len_x = var[0].shape
    y_vals = np.arange(0, len_y, y_stride)
    x_vals = np.arange(0, len_x, x_stride)
    outputs = {'t0': [], 't1': [], 't2': []}

    for y in y_vals:
        for x in x_vals:
            current_output = var[..., y:y+y_stride,x:x+x_stride]

            if current_output.shape[-2:] == (y_stride, x_stride):
                if not np.isnan(np.sum(current_output)):
                    outputs['t0'].append(current_output[0])
                    outputs['t1'].append(current_output[1])
                    outputs['t2'].append(current_output[2])

    return np.array(list(outputs.values()), dtype=np.float32)

def get_x_y_patch(var_x, var_y, x_idx, y_idx, stride=250):
    outputs_x = []
    outputs_y = []
    for y,x in zip(y_idx, x_idx):
        xx = var_x[x:x+stride]
        yy = var_y[y:y+stride]
        x_mesh, y_mesh = np.meshgrid(xx,yy)
        outputs_x.append(x_mesh)
        outputs_y.append(y_mesh)

    return np.stack(outputs_x), np.stack(outputs_y)

def get_valid_patches(var, x_idx, y_idx, stride=250):
    new_x_idx = []
    new_y_idx = []
    for y,x in zip(y_idx, x_idx):
        current_output = var[..., y:y+stride,x:x+stride]
        
        if current_output.shape[-2:] == (stride, stride):
            if not np.isnan(np.sum(current_output)):
                new_x_idx.append(x)
                new_y_idx.append(y)

    return new_x_idx, new_y_idx


def sliding_window_from_idx(var, x_idx, y_idx, stride=250):      # 250km x 250km grid
    outputs = {'t0': [], 't1': [], 't2': []}
    for y,x in zip(y_idx, x_idx):

        current_output = var[..., y:y+stride,x:x+stride]

        outputs['t0'].append(current_output[0])
        outputs['t1'].append(current_output[1])
        outputs['t2'].append(current_output[2])

    return np.array(list(outputs.values()), dtype=np.float32)


def determine_meanSIC(sic, x_stride = 250, y_stride = 250):
    """Function to determine meanSIC in all cells
        Uses a bottom up approach when sliding the window

    Args:
        sic (np.array(1, 2784, 2694)): SIC chart with AROME borders
    """

    len_y, len_x = sic[0].shape
    y_vals = np.arange(0, len_y, y_stride)
    x_vals = np.arange(0, len_x, x_stride)

    outputs = {'below': {'x': [], 'y': [], 'mean_sic': []}, 
               'between': {'x': [], 'y': [], 'mean_sic': []},
               'above': {'x': [], 'y': [], 'mean_sic': []}}


    for y in y_vals:
        for x in x_vals:
            current_output = sic[0, y:y+y_stride, x:x+x_stride]

            if not np.ma.is_masked(current_output):
                mean = current_output.mean()
                if mean <= 10.:
                    pos = 'below'
                elif mean >= 90.:
                    pos = 'above'
                else:
                    pos = 'between'
                    
                outputs[pos]['x'].append(x)
                outputs[pos]['y'].append(y)
                outputs[pos]['mean_sic'].append(mean)

    return outputs
    

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx 

def runstuff():
    # Setup data
    path_IceChart = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    paths = []
    for year in range(2019, 2022):
    # for year in range(2019, 2020): # Only want one year
        for month in range(1, 13):
        # for month in range(1, 2): # Only want one month
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)

    #
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    hdf5_path = f"{path_output}/FullPatchedAromeArctic.hdf5"
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    outfile = h5py.File(hdf5_path, "a")

    for path_data_task in paths:
        print(f"path_data_task = {path_data_task}")
        year_task = path_data_task[len(path_data) : len(path_data) + 4]
        print(f"year_task = {year_task}")
        month_task = path_data_task[len(path_data) + 5 : len(path_data) + 7]
        print(f"month_task = {month_task}")
        nb_days_task = monthrange(int(year_task), int(month_task))[1]
        print(f"nb_days_task = {nb_days_task}")
        #
        if os.path.isdir(path_output) == False:
            os.system('mkdir -p ' + path_output)


        for dd in range(1, nb_days_task + 1):
            yyyymmdd = f"{year_task}{month_task}{dd:02d}"
            print(f"{yyyymmdd}")

            try:
                arome_path = glob.glob(f"{path_data_task}AROME_ICgrid_{yyyymmdd}T00Z.nc")[0]
                ic_path = glob.glob(f"{path_IceChart}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

            except IndexError:
                continue

            nc = Dataset(arome_path, 'r')
            x = nc.variables['xc']
            y = nc.variables['yc']
            temp = nc.variables['T2M']
            sst = nc.variables['SST']
            xwind = nc.variables['X_wind_10m']
            ywind = nc.variables['Y_wind_10m']

            nc_IC = Dataset(ic_path, 'r')

            xc = nc_IC['xc'][:]
            yc = nc_IC['yc'][:]

            xmin = find_nearest(xc, x[:].min())
            xmax = find_nearest(xc, x[:].max())
            ymin = find_nearest(yc, y[:].min())
            ymax = find_nearest(yc, y[:].max())

            SIC_thresholds = determine_meanSIC(nc_IC['ice_concentration'][..., ymin:ymax, xmin:xmax])

            nc_IC.close()

            for key in SIC_thresholds.keys():
                x_idx = np.array(SIC_thresholds[key]['x'])
                y_idx = np.array(SIC_thresholds[key]['y'])
                x_idx, y_idx = get_valid_patches(temp, x_idx, y_idx)

                try:
                    xc, yc = get_x_y_patch(x, y, x_idx, y_idx)
                except ValueError:
                    for key in SIC_thresholds.keys():
                        offending = f"{key}/{yyyymmdd}"
                        if offending in outfile:
                            del outfile[offending]

                    break

                outfile[f"{key}/{yyyymmdd}/xc"] = xc
                outfile[f"{key}/{yyyymmdd}/yc"] = yc
                outfile[f"{key}/{yyyymmdd}/temp"] = sliding_window_from_idx(temp, x_idx, y_idx)
                outfile[f"{key}/{yyyymmdd}/sst"] = sliding_window_from_idx(sst, x_idx, y_idx)
                outfile[f"{key}/{yyyymmdd}/xwind"] = sliding_window_from_idx(xwind, x_idx, y_idx)
                outfile[f"{key}/{yyyymmdd}/ywind"] = sliding_window_from_idx(ywind, x_idx, y_idx)

            nc.close()


    outfile.close()

if __name__ == "__main__":
    runstuff()

#################################################
EOF

python3 "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/data_processing_files/PROG/FULL_Patch_to_hdf5.py"
