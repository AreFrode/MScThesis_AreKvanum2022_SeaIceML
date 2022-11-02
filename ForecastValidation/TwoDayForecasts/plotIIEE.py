import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')

import os
import glob
import h5py

import numpy as np

from verification_metrics import IIEE

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from shapely.errors import ShapelyDeprecationWarning

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


def main():
    model_name = "unet_benchmark"

    PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/{model_name}/"
    PATH_FIGURES = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/figures/{model_name}/"

    icecharts = sorted(glob.glob(f"{PATH_PERSISTANCE}2021/**/*.hdf5", recursive = True))
    forecasts = sorted(glob.glob(f"{PATH_FORECAST}2021/**/*.hdf5", recursive = True))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][450:, :1840]
        lat = constants['lat'][450:, :1840]
        lon = constants['lon'][450:, :1840]

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()


    for target, forecast in zip(icecharts, forecasts):
        date = forecast[-17:-9]
        print(date)

        save_location = f"{PATH_FIGURES}{date[:4]}/{date[4:6]}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        elif os.path.exists(f"{save_location}{date}.png"):
            continue

        with h5py.File(target, 'r') as infile:
            sic_target = infile['sic_target'][450:, :1840]

        with h5py.File(forecast, 'r') as infile:
            sic_forecast = infile['y_pred'][0]

        iiee = IIEE(sic_forecast, sic_target, lsmask)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_title(f"IIEE for two day forecast initiated {date}", fontsize=30)
        ax.set_extent([-20, 45, 60, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

        ax.pcolormesh(lon, lat, np.ma.masked_array(iiee[0], iiee[0] < 1), transform=data_proj, zorder=2, cmap=plt.colormaps['Greens_r'])
        ax.pcolormesh(lon, lat, np.ma.masked_array(iiee[1], iiee[1] < 1), transform=data_proj, zorder=2, cmap=plt.colormaps['Reds_r'])
        ax.pcolormesh(lon, lat, np.ma.masked_array(iiee[2], iiee[2] < 1), transform=data_proj, zorder=2, cmap=plt.colormaps['cividis'])
        ax.pcolormesh(lon, lat, np.ma.masked_array(iiee[3], iiee[3] < 1), transform=data_proj, zorder=2, cmap=plt.colormaps['cividis_r'])

        plt.savefig(f"{save_location}{date}.png")

        ax.cla()
        plt.close(fig)
        del iiee


if __name__ == "__main__":
    main()