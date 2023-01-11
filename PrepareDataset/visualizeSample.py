import h5py
import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast")

import LambertLabels

import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from shapely.errors import ShapelyDeprecationWarning
from pyproj import Proj

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

def add_water(newcolors):
    newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    newcolors[0, -1] = 0.3
    return newcolors


def main():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/osisaf_trend_5/2022/01/PreparedSample_20220103.hdf5"

    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/figures/sample_visualize/"

    map_proj = ccrs.LambertConformal(central_latitude= 77.5, central_longitude = -25, standard_parallels = (77.5, 77.5))

    PRJ = Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    lower_boundary = 578
    rightmost_boundary = 1792

    with h5py.File(path, 'r') as f:
        lat = f['lat'][lower_boundary:, :rightmost_boundary]
        lon = f['lon'][lower_boundary:, :rightmost_boundary]

        lsmask = f['lsmask'][lower_boundary:, :rightmost_boundary]
        sic = f['sic'][lower_boundary:,     :rightmost_boundary]
        sic_target = f['sic_target'][lower_boundary:, :rightmost_boundary]
        sic_trend = f['sic_trend'][lower_boundary:, :rightmost_boundary]
        t2m = f['t2m'][lower_boundary:, :rightmost_boundary]
        xwind = f['xwind'][lower_boundary:, :rightmost_boundary]
        ywind = f['ywind'][lower_boundary:, :rightmost_boundary]

    sic_trend = np.ma.masked_where(sic_trend == 0.0, sic_trend)

    datas = [sic, sic_target, lsmask, sic_trend, t2m, xwind, ywind]
    names = ['sic', 'sic_target', 'lsmask', 'sic_trend', 't2m', 'xwind', 'ywind']

    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    yticks = [60,65,70, 75, 80, 85,90]

    cividis = mpl.colormaps['cividis']
    newcolors = cividis(np.linspace(0, 1, 7))
    newcolors = add_water(newcolors)
    ice_cmap = colors.ListedColormap(newcolors)

    ice_levels = np.linspace(0, 7, 8, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    lsmask_color = mpl.colormaps['Wistia']
    newcolors = lsmask_color(np.linspace(0,1,2))
    newcolors = add_water(newcolors)
    lsmask_cmap = colors.ListedColormap(newcolors)


    RYG = mpl.colormaps['RdYlGn']
    coolwarm = mpl.colormaps['coolwarm']
    winter = mpl.colormaps['winter']

    cmaps = [ice_cmap, ice_cmap, lsmask_cmap, RYG, coolwarm, winter, winter]
    norms = [ice_norm, ice_norm, None, None, None, None, None]
    

    for data, name, cmap, norm in zip(datas, names, cmaps, norms):
        print(name)
        fig = plt.figure(figsize=(20,20))
        # fig.subplots_adjust(bottom = 0.2)
        ax = plt.axes(projection=map_proj)
    
        ax.pcolormesh(lon, lat, data, transform=data_proj, norm = norm, cmap = cmap, zorder=1)

        ax.set_xlim(x0,x1)
        ax.set_ylim(y0,y1)

        fig.canvas.draw()
        ax.gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        LambertLabels.lambert_xticks(ax, xticks)
        LambertLabels.lambert_yticks(ax, yticks)
        ax.add_feature(cfeature.COASTLINE, lw=2)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{path_figures}{name}.png", bbox_inches = 'tight', pad_inches = 0)

        ax.cla()
        plt.close(fig)





if __name__ == "__main__":
    main()