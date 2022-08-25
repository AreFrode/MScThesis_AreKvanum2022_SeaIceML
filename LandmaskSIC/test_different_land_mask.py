import glob
import os
import numpy as np

import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import pyplot as plt
from calendar import monthrange
from scipy.ndimage import distance_transform_edt
from netCDF4 import Dataset

# This script applies a land mask as outlined in Wang2017
# The land mask is a simple nearest neighbor padding of the SIC over land


def runstuff():
    path_data = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_output = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/Data/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/LandmaskSIC/figures/"

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    paths = []
    # for year in range(2019, 2022):
    for year in range(2019, 2020): # Only want one year
        # for month in range(1, 13):
        for month in range(1, 2): # Only want one month
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)

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

    for dd in range(2, 3):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")

        try:
            ic_path = glob.glob(f"{path_data}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

        except IndexError:
            continue

        output_filename = f"ice_conc_svalbard_landmask_{yyyymmdd}1500.nc"

        icechart = Dataset(ic_path)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)

        iceconc1 = icechart['ice_concentration'][0,:,:]
        iceconc1 = iceconc1.filled(-100)
    
        pcolor1 = ax.pcolormesh(icechart['lon'], icechart['lat'], iceconc1, transform=data_proj)

        plt.savefig(f"{path_figures}iceconc_valueoor.png")

        iceconc2 = icechart['ice_concentration'][0,:,:]
        iceconc2 = iceconc2.filled(np.nan)

        indices = distance_transform_edt(np.isnan(iceconc2), return_distances=False, return_indices=True)
        iceconc2 = iceconc2[tuple(indices)]

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)

        pcolor2 = ax.pcolormesh(icechart['lon'], icechart['lat'], iceconc2, transform=data_proj)


        plt.savefig(f"{path_figures}iceconc_mirrornn.png")

        icechart.close()



if __name__ == "__main__":
    runstuff()