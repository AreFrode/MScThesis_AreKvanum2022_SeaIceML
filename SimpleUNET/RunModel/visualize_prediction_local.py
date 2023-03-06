import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")

import h5py
import glob
import os
import LambertLabels
import pyproj

import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from datetime import datetime, timedelta
from helper_functions import read_config_from_csv

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/01/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/figures/"

    
    data_2022 = np.array(sorted(glob.glob(f"{path_pred}{weights}/2022/**/*.hdf5")))

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[0]

    with h5py.File(h5file, 'r') as f:
        lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
        lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    yticks = [60,65,70, 75, 80, 85,90]

    cividis = mpl.colormaps['cividis']

    preferred_cmap = cividis(np.linspace(0, 1, config['num_outputs']))

    newcolors = cividis(np.linspace(0, 1, 6))
    newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    newcolors[0, -1] = 0.3
    newcolors[1:] = preferred_cmap[2:]
    ice_cmap = colors.ListedColormap(newcolors)

    ice_levels = np.linspace(0, config['num_outputs'] - 1, config['num_outputs'], dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    if config['reduced_classes']:
        ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100']

    else:
        ice_ticks = ['0', '0 - 10', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']

    ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']

    for date in data_2022:
        # New definition after name change
        yyyymmdd_b = date[-17:-9]
        yyyymmdd_v = date[-27:-19]

        year = yyyymmdd_b[:4]
        month = yyyymmdd_b[4:6]

        save_location = f"{path_figures}{weights}/{year}/{month}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        f_model = h5py.File(f"{path_pred}{weights}/{year}/{month}/SIC_UNET_v{yyyymmdd_v}_b{yyyymmdd_b}T15Z.hdf5", 'r')

        y_pred = f_model['y_pred'][0]

        y_pred = np.where(y_pred == 1, 0, y_pred)
        y_pred = np.where(y_pred > 0, y_pred - 1, y_pred)

        # Plotting

        # cmap = plt.get_cmap('cividis', 7)

        fig = plt.figure(figsize=(20,20))
        # fig.subplots_adjust(bottom = 0.2)
        ax = plt.axes(projection=map_proj)
        ax.set_title(f"Forecast for {yyyymmdd_v} initiated {yyyymmdd_b}", fontsize = 30)
    
        ax.pcolormesh(lon, lat, y_pred, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1)
        ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap='autumn')

        # cbar_ax = fig.add_axes([0.15, 0.1, 0.6, 0.025])
        mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)
        # mapper.set_array([-1, 8])

        cbar = fig.colorbar(mapper,
                            ax = ax,
                            spacing = 'uniform',
                            location = 'bottom',
                            orientation = 'horizontal',
                            shrink = .7,
                            pad = .05
        )

        cbar.set_label(label = 'SIC range [%]', size = 16)
        cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)
        cbar.ax.tick_params(labelsize = 16)

        ax.set_xlim(x0,x1)
        ax.set_ylim(y0,y1)

        fig.canvas.draw()
        ax.gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        LambertLabels.lambert_xticks(ax, xticks)
        LambertLabels.lambert_yticks(ax, yticks)


        plt.savefig(f"{save_location}alt_v{yyyymmdd_v}_b{yyyymmdd_b}.png")

        f_model.close()
        ax.cla()
        plt.close(fig)


if __name__ == "__main__":
    main()