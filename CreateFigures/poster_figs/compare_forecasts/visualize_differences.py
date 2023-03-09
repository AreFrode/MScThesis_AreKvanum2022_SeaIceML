import sys
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")

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
from netCDF4 import Dataset
from createHDF import onehot_encode_sic
from scipy.interpolate import NearestNDInterpolator


from datetime import datetime, timedelta
from helper_functions import read_config_from_csv

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    path_raw = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"

    path = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/01/"
    path_figures = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/poster_figs/"

    lower_boundary = 578
    rightmost_boundary = 1792

    year = '2022'
    month = '03'
    bday = '23'
    vday = '25'

    data_ml = glob.glob(f"{path_pred}{weights}/{year}/{month}/SIC_UNET_v{year}{month}{vday}_b{year}{month}{bday}T15Z.hdf5")[0]

    data_ict0 = glob.glob(f"{path_raw}{year}/{month}/ICECHART_1kmAromeGrid_{year}{month}{bday}T1500Z.nc")[0]
    data_ict2 = glob.glob(f"{path_raw}{year}/{month}/ICECHART_1kmAromeGrid_{year}{month}{vday}T1500Z.nc")[0]


    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

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

    fig = plt.figure(figsize=(17,25.5), constrained_layout = True)
    fig.canvas.draw()
    axs = fig.subplot_mosaic('''
                             abc
                             ''', 
                             subplot_kw={"projection": map_proj}
                             )

    with h5py.File(sorted(glob.glob(f"{path}*.hdf5"))[0], 'r') as f:
        lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
        lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:622, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    # Plot figure a
    with Dataset(data_ict0, 'r') as ic_0:
        sic0 = onehot_encode_sic(ic_0['sic'][lower_boundary:, :rightmost_boundary])
    
    sic0_interpolator = NearestNDInterpolator(mask_T, sic0[mask])
    sic0 = sic0_interpolator(*np.indices(sic0.shape))

    sic0 = np.where(sic0 == 1, 0, sic0)
    sic0 = np.where(sic0 > 0, sic0 - 1, sic0)

    axs['a'].pcolormesh(lon, lat, sic0, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1)
    axs['a'].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap='autumn')

    axs['a'].set_xlim(x0,x1)
    axs['a'].set_ylim(y0,y1)

        
    axs['a'].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    axs['a'].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    axs['a'].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(axs['a'], xticks)
    LambertLabels.lambert_yticks(axs['a'], yticks)

    mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)
        # mapper.set_array([-1, 8])

    cbar = fig.colorbar(mapper,
                        ax = axs['a'],
                        spacing = 'uniform',
                        location = 'bottom',
                        orientation = 'horizontal',
                        shrink = .7,
                        pad = .05
    )

    cbar.set_label(label = 'SIC range [%]', size = 16)
    cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)
    cbar.ax.tick_params(labelsize = 16)





    plt.show()
    exit()

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