import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")


import h5py
import glob
import os
import LambertLabels
import pyproj
import WMOcolors

import numpy as np
import matplotlib.colors as colors

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from datetime import datetime
from helper_functions import read_config_from_csv
from netCDF4 import Dataset
from verification_metrics import find_ice_edge

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/01/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/forecast_time_series/"

    osisaf_iceedge = sorted(glob.glob("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSI_SAF_regrid/Data/old/2022/**/*.nc"))
    
    data_2022 = []
    for m in range(1, 13):
        data_2022.append(sorted(glob.glob(f"{path_pred}{weights}/2022/{m:02d}/*.hdf5"))[0])

    data_2022 = np.array(data_2022).reshape(4,3)
    titles = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Des']).reshape(4,3)


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

    xticks = []
    yticks = []

    # cividis = mpl.colormaps['cividis']
    # newcolors = cividis(np.linspace(0, 1, config['num_outputs']))
    # newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    # newcolors[0, -1] = 0.3
    # ice_cmap = colors.ListedColormap(newcolors)
    ice_cmap = WMOcolors.cm.sea_ice_chart()
    land_cmap = WMOcolors.cm.land()

    ice_levels = np.linspace(0, config['num_outputs'], config['num_outputs'] + 1, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    if config['reduced_classes']:
        ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100']

    else:
        ice_ticks = ['0', '0 - 10', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']


    save_location = f"{path_figures}"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    
    # Plotting

    # cmap = plt.get_cmap('cividis', 7)

    nrows = 4
    ncols = 3

    fig, axs = plt.subplots(nrows= nrows, ncols = ncols, subplot_kw={'projection': map_proj}, figsize=(15, 20))
    # fig.subplots_adjust(bottom = 0.2)
    # ax = plt.axes(projection=map_proj)
    # ax.set_title(f"Forecast for {yyyymmdd_v} initiated {yyyymmdd_b}", fontsize = 30)

    for i in range(nrows):
        for j in range(ncols):
            print(f"Plotting row: {i}, col: {j}")
            with h5py.File(data_2022[i,j], 'r') as f_model:
                y_pred = f_model['y_pred'][0]

            doy= datetime.strptime(data_2022[i,j][-27:-19], '%Y%m%d').timetuple().tm_yday

            with Dataset(osisaf_iceedge[doy], 'r') as nc:
                sic = nc.variables['ice_conc'][0, 578:, :1792]

            day_ice_edge = find_ice_edge(sic, lsmask)


            axs[i,j].pcolormesh(lon, lat, y_pred, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1)
            axs[i,j].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap=land_cmap)
            axs[i,j].scatter(lon, lat, s=.1*day_ice_edge, zorder = 3, transform = data_proj, c='black', alpha=.5)
            # cbar_ax = fig.add_axes([0.15, 0.1, 0.6, 0.025])


            axs[i,j].set_xlim(x0,x1)
            axs[i,j].set_ylim(y0,y1)

            axs[i,j].set_title(f"{data_2022[i,j][-27:-19][-2:]} {titles[i,j]} 2022")

            # fig.canvas.draw()
            # axs[i,j].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
            # axs[i,j].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            # axs[i,j].yaxis.set_major_formatter(LATITUDE_FORMATTER)
            # LambertLabels.lambert_xticks(axs[i,j], xticks)
            # LambertLabels.lambert_yticks(axs[i,j], yticks)


    # mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)
    # mapper.set_array([-1, 8])

    # cbar = fig.colorbar(mapper,
                        # ax = ax,
                        # spacing = 'uniform',
                        # location = 'bottom',
                        # orientation = 'horizontal',
                        # shrink = .7,
                        # pad = .05
    # )

    # cbar.set_label(label = 'SIC range [%]', size = 16)
    # cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)
    # cbar.ax.tick_params(labelsize = 16)

    
    plt.savefig(f"{save_location}Forecast_time_series.png")


if __name__ == "__main__":
    main()