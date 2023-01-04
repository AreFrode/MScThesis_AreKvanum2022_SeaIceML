import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")

import h5py
import glob
import os

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER

from datetime import datetime, timedelta
from helper_functions import read_config_from_csv

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/osisaf_trend_{config['osisaf_trend']}/2022/01/"
    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/figures/"

    
    data_2022 = np.array(sorted(glob.glob(f"{path_pred}{weights}/2022/**/*.hdf5")))

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[0]

    f = h5py.File(h5file, 'r')
    lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
    lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
    lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    for date in data_2022:
        yyyymmdd = date[-17:-9]
        print(f"{yyyymmdd}")
        year = yyyymmdd[:4]
        month = yyyymmdd[4:6]

        init_date = datetime.strptime(yyyymmdd, '%Y%m%d')
        init_date = (init_date - timedelta(days = 2)).strftime('%Y%m%d')

        save_location = f"{path_figures}{weights}/{year}/{month}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        f_model = h5py.File(f"{path_pred}{weights}/{year}/{month}/SIC_SimpleUNET_two_day_forecast_{yyyymmdd}T15Z.hdf5", 'r')

        y_pred = f_model['y_pred'][0]

        cmap = plt.get_cmap('cividis', 7)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_title(f"Forecast for two day forecast initiated {init_date}", fontsize = 30)
        ax.set_extent([-18, 45, 65, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

        gl = ax.gridlines(
            crs=data_proj, 
            draw_labels=True,
            linewidth=2,
            color='snow',
            alpha=.7,
            linestyle='--',
            zorder = 5
        )

        gl.xlines = False
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.left_labels = False
        gl.ylabel_style = {'size': 25}

        cbar = ax.pcolormesh(lon, lat, y_pred, transform=data_proj, zorder=2, cmap=cmap)
        ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap='autumn')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, map_projection=ccrs.PlateCarree())
        plt.colorbar(cbar, cax = cax)

        plt.savefig(f"{save_location}{yyyymmdd}.png")

        f_model.close()
        ax.cla()
        plt.close(fig)

    f.close()


if __name__ == "__main__":
    main()