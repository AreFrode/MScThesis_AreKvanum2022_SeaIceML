import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")

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
from createHDF import onehot_encode_sic
from scipy.interpolate import NearestNDInterpolator


from netCDF4 import Dataset

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    path_icecharts = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"

    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/figures/"

    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"


    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    
    data_2022 = sorted(glob.glob(f"{path_icecharts}/2022/**/*.nc"))

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    constants = data_2022[0]

    lower_boundary = 578
    rightmost_boundary = 1792
    num_outputs = 6
    

    with Dataset(f"{path_arome}2019/01/AROME_1kmgrid_20190101T18Z.nc") as fa:
        lsmask = fa['lsmask'][lower_boundary:,:rightmost_boundary]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:622, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    with Dataset(constants, 'r') as f:
        lat = f['lat'][lower_boundary:, :rightmost_boundary]
        lon = f['lon'][lower_boundary:, :rightmost_boundary]

    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    yticks = [60,65,70, 75, 80, 85,90]

    cividis = mpl.colormaps['cividis']
    preferred_cmap = cividis(np.linspace(0, 1, num_outputs + 1))

    newcolors = cividis(np.linspace(0, 1, num_outputs))
    newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    newcolors[0, -1] = 0.3
    newcolors[1:] = preferred_cmap[2:]
    ice_cmap = colors.ListedColormap(newcolors)

    ice_levels = np.linspace(0, num_outputs, num_outputs + 1, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    if num_outputs == 5:
        ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100']
    
    if num_outputs == 6:
        ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']

    else:
        ice_ticks = ['0', '0 - 10', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']

    for data in data_2022:
        # New definition after name change
        yyyymmdd_v = data[-17:-9]

        year = yyyymmdd_v[:4]
        month = yyyymmdd_v[4:6]

        save_location = f"{path_figures}/{year}/{month}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)


        with Dataset(data, 'r') as ic:
            sic = onehot_encode_sic(ic['sic'][lower_boundary:, :rightmost_boundary])

        sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
        sic = sic_interpolator(*np.indices(sic.shape))

        sic = np.where(sic == 1, 0, sic)
        sic = np.where(sic > 0, sic - 1, sic)

        # Plotting

        # cmap = plt.get_cmap('cividis', 7)

        fig = plt.figure(figsize=(20,20))
        # fig.subplots_adjust(bottom = 0.2)
        ax = plt.axes(projection=map_proj)
        ax.set_title(f"Icechart {yyyymmdd_v}", fontsize = 30)
    
        ax.pcolormesh(lon, lat, sic, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1)
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


        plt.savefig(f"{save_location}v{yyyymmdd_v}.png")

        ax.cla()
        plt.close(fig)


if __name__ == "__main__":
    main()