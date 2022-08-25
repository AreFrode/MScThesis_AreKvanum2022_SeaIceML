import glob
import h5py
import os
import numpy as np
from calendar import monthrange
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt

# The point of this script is to discretize the SIC into some premade bins, as this might simplify training as the model doesnt have to predict numerical values but classes instead. Also might create several small hdf5 files instead of one large hdf5 files, as in line with MET conventions

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

def sliding_window_no_time(var, x_idx, y_idx, stride=250):
    outputs = []
    for y,x in zip(y_idx, x_idx):
        current_output = var[y:y+stride,x:x+stride]
        outputs.append(current_output)

    return np.array(outputs, dtype=np.float32)

def sliding_window_from_idx(var, x_idx, y_idx, stride=250):      # 250km x 250km grid
    outputs = {'t0': [], 't1': [], 't2': []}
    for y,x in zip(y_idx, x_idx):

        current_output = var[..., y:y+stride,x:x+stride]

        outputs['t0'].append(current_output[0])
        outputs['t1'].append(current_output[1])
        outputs['t2'].append(current_output[2])

    return np.array(list(outputs.values()), dtype=np.float32)

def icechart_patch_from_idx_onehot_encoded(ic, x_idx, y_idx, stride=250):
    outputs = []
    for y,x in zip(y_idx, x_idx):
        outputs.append([])
        icepatch = ic[y:y+stride,x:x+stride]
        # icepatch = icepatch.filled(np.nan)

        # indices = distance_transform_edt(np.isnan(icepatch), return_distances=False, return_indices=True)
        # icepatch = icepatch[tuple(indices)]

        n,m = icepatch.shape

        for i in range(n):
            for j in range(m):
                val = icepatch[i,j]
                if val < 50.:
                    outputs[-1].append([1,0])
                else:
                    outputs[-1].append([0,1])

    return np.array(outputs, dtype=np.int8)

def determine_meanSIC(sic, x_stride = 250, y_stride = 250):
    """Function to determine meanSIC in all cells
        Uses a bottom up approach when sliding the window

    Args:
        sic (np.array(1, 2784, 2694)): SIC chart with AROME borders
    """

    len_y, len_x = sic[0].shape
    y_vals = np.arange(0, len_y, y_stride)
    x_vals = np.arange(0, len_x, x_stride)

    outputs = {'below': {'x': [], 'y': []}, 
               'between': {'x': [], 'y': []},
               'above': {'x': [], 'y': []}}


    for y in y_vals:
        for x in x_vals:
            current_output = sic[0, y:y+y_stride, x:x+x_stride]

            mean = np.ma.mean(current_output)

            if mean.dtype == np.dtype(np.float64):
                if mean <= 10.:
                    pos = 'below'
                elif mean >= 90.:
                    pos = 'above'
                else:
                    pos = 'between'
                    
                outputs[pos]['x'].append(x)
                outputs[pos]['y'].append(y)

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
        for month in range(1, 13):
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)


    # path_data_task = paths[$SGE_TASK_ID - 1]
    path_data_task = paths[0]
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
            arome_path = glob.glob(f"{path_data_task}AROME_ICgrid_{yyyymmdd}T00Z.nc")[0]
            ic_path = glob.glob(f"{path_IceChart}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

        except IndexError:
            continue

        hdf5_path = f"{path_output}{year_task}/{month_task}/PatchedAromeArcticBins_{yyyymmdd}.hdf5"
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        outfile = h5py.File(hdf5_path, "a")

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

        lat = nc_IC['lat'][:]
        lon = nc_IC['lon'][:]

        xmin = find_nearest(xc, x[:].min())
        xmax = find_nearest(xc, x[:].max())
        ymin = find_nearest(yc, y[:].min())
        ymax = find_nearest(yc, y[:].max())

        sic = nc_IC['ice_concentration'][..., ymin:ymax, xmin:xmax]

        SIC_thresholds = determine_meanSIC(sic)

        for key in SIC_thresholds.keys():
            x_idx = np.array(SIC_thresholds[key]['x'])
            y_idx = np.array(SIC_thresholds[key]['y'])
            x_idx, y_idx = get_valid_patches(temp, x_idx, y_idx)

            try:
                xc, yc = get_x_y_patch(x, y, x_idx, y_idx)
            except ValueError:
                # Kind of extreme, but delete file when error occurs
                
                os.remove(hdf5_path)
                break

            outfile[f"{key}/xc"] = xc
            outfile[f"{key}/yc"] = yc
            outfile[f"{key}/lat"] = sliding_window_no_time(lat, x_idx, y_idx)
            outfile[f"{key}/lon"] = sliding_window_no_time(lon, x_idx, y_idx)
            outfile[f"{key}/t2m"] = sliding_window_from_idx(temp, x_idx, y_idx)
            outfile[f"{key}/sst"] = sliding_window_from_idx(sst, x_idx, y_idx)
            outfile[f"{key}/xwind"] = sliding_window_from_idx(xwind, x_idx, y_idx)
            outfile[f"{key}/ywind"] = sliding_window_from_idx(ywind, x_idx, y_idx)
            outfile[f"{key}/sic"] = icechart_patch_from_idx_onehot_encoded(sic[0,...], x_idx, y_idx)

        nc_IC.close()
        nc.close()

        outfile.close()

        break

if __name__ == "__main__":
    runstuff()