import glob
import h5py
import os

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def extrapolate_border(field):
    mask = np.where(~np.isnan(field))
    interp = NearestNDInterpolator(np.transpose(mask), field[mask])
    return interp(*np.indices(field.shape))


def runstuff():
    # Setup data-paths
    path_IceChart = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_RegridArome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/testingdata/" # TODO subject to change
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    paths = []
    for year in range(2019, 2022):
        for month in range(1,13):
            p = f"{path_RegridArome}{year}/{month:02d}/"
            paths.append(p)

    # path_data_task = paths[$SGE_TASK_ID - 1]
    path_data_task = paths[0]
    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_RegridArome) : len(path_RegridArome) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_RegridArome) + 5 : len(path_RegridArome) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")

    if not os.path.isdir(f"{path_output}{year_task}/{month_task}"):
        os.makedirs(f"{path_output}{year_task}/{month_task}")

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")

        try:
            arome_path = glob.glob(f"{path_data_task}/AROME_ICgrid_{yyyymmdd}T00Z.nc")[0]
            ic_path = glob.glob(f"{path_IceChart}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

        except IndexError:
            continue

        hdf5_path = f"{path_output}{year_task}/{month_task}/PreparedExtrapolated_{yyyymmdd}.hdf5"
        
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        outfile = h5py.File(hdf5_path, "a")

        nc = Dataset(arome_path, 'r')
        lat = nc.variables['lat']
        lon = nc.variables['lon']

        t2m = nc.variables['T2M']
        sst = nc.variables['SST']
        xwind = nc.variables['X_wind_10m']
        ywind = nc.variables['Y_wind_10m']
        windspeed = nc.variables['10WindSpeed']
        winddir = nc.variables['10WindDirection']
        lsmask = nc.variables['LSMASK']
        oobmask = nc.variables['OutOfBoundsMask']

        x = nc.variables['x']
        y = nc.variables['y']

        x_min = x[:].min()
        x_max = x[:].max()
        y_min = y[:].min()
        y_max = y[:].max()

        nc_IC = Dataset(ic_path, 'r')
        xc = nc_IC.variables['xc'][:]
        yc = nc_IC.variables['yc'][:]

        xc_min = find_nearest(xc, x_min)
        xc_max = find_nearest(xc, x_max)
        yc_min = find_nearest(yc, y_min)
        yc_max = find_nearest(yc, y_max)

        sic = nc_IC['ice_concentration'][..., yc_min:yc_max, xc_min:xc_max]

        outfile[f"lat"] = lat[:]
        outfile[f"lon"] = lon[:]
        outfile[f"sic"] = sic[0]
        outfile[f"sst"] = extrapolate_border(sst[0])
        outfile[f"lsmask"] = extrapolate_border(lsmask[0])
        outfile[f"oobmask"] = oobmask[0]

        for day in range(3):   # Arome Regrid contain 3 days
            key = f"day{day}"

            outfile[f"{key}/t2m"] = extrapolate_border(t2m[day])        


        nc_IC.close()
        nc.close()
        outfile.close()

if __name__ == "__main__":
    runstuff()