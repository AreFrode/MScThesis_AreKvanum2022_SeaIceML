import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")


import LambertLabels
import WMOcolors

import numpy as np
import matplotlib as mpl
import pyproj

from netCDF4 import Dataset
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import path as mpath
from matplotlib import patches as mpatches
from matplotlib import ticker as mticker
from cartopy.mpl import ticker as ctk
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from shapely.errors import ShapelyDeprecationWarning
from createHDF import onehot_encode_sic


import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 



def main():
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/09/ICECHART_1kmAromeGrid_20220915T1500Z.nc"

    bottom = 578
    right = 1792

    with Dataset(PATH_DATA) as nc:
        lat = nc.variables['lat'][bottom:, :right]
        lon = nc.variables['lon'][bottom:, :right]
        sic = onehot_encode_sic(nc.variables['sic'][bottom:, :right])



    baltic_mask = np.zeros_like(sic)
    baltic_mask[:400, 1400:] = 1

    sic = np.where(baltic_mask == 1, 0, sic)

    map_proj = ccrs.LambertConformal(central_longitude = 20)
    map_proj.threshold = 1e3

    data_proj = ccrs.PlateCarree()
    
    PRJ = pyproj.Proj(map_proj.proj4_init)

    xticks = [-60, -40, -20, 0, 20, 40, 60, 80, 100]
    yticks = [60, 80]

    # Set colormap
    cividis = mpl.colormaps['cividis']
    newcolors = cividis(np.linspace(0, 1, 7))
    newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    newcolors[0, -1] = 0.3
    # ice_cmap = colors.ListedColormap(newcolors)
    ice_cmap = WMOcolors.cm.sea_ice_chart()
    

    LAND_highres = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        facecolor=cfeature.COLORS['land'],
                                        edgecolor='black',
                                        linewidth=.7,
                                        zorder = 2)

    fig = plt.figure(facecolor='w', edgecolor='k')
    ax = plt.axes(projection = map_proj)
    gl = ax.gridlines(crs = data_proj, draw_labels = True, x_inline = False, y_inline = False, color = 'dimgray', linestyle = (1, (1, 7.5)), lw = 1.2, rotate_labels = False, xlocs = mticker.FixedLocator(xticks), ylocs = mticker.FixedLocator(yticks))

    gl.top_labels = False
    gl.right_labels = False

    # gl.xformatter = ctk.LongitudeFormatter(zero_direction_label = True)
    # gl.yformatter = ctk.LatitudeFormatter()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
  
    ax.add_feature(LAND_highres)

    ax.pcolormesh(lon, lat, sic, transform = data_proj, cmap = ice_cmap, zorder = 0, rasterized = True)

    xlim = [-60, 100]
    ylim = [50, 89]

    rect = mpath.Path([[xlim[0], ylim[0]],
                   [xlim[1], ylim[0]],
                   [xlim[1], ylim[1]],
                   [xlim[0], ylim[1]],
                   [xlim[0], ylim[0]],
                   ])

    proj_to_data = data_proj._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)

    ax.add_patch(mpatches.PathPatch(rect_in_target, facecolor='None', lw=4, zorder =2))
    ax.set_boundary(rect_in_target)
    ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=data_proj)

    ax.plot(lon[:,0],lat[:,0],'--', color = 'black', lw=1.5, transform=ccrs.PlateCarree(), zorder = 1)
    ax.plot(lon[:,-1],lat[:,-1],'--', color = 'black', lw=1.5, transform=ccrs.PlateCarree(), zorder = 1)
    ax.plot(lon[0,:],lat[0,:],'--', color = 'black', lw = 1.5, transform=ccrs.PlateCarree(), zorder = 1)
    ax.plot(lon[-1,:],lat[-1,:],'--', color = 'black', lw = 1.5, transform=ccrs.PlateCarree(), zorder = 1)

    ice_levels = np.linspace(0, 7, 8, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)
    ice_ticks = ['ice free', '<10', '10–30', '40–60', '70–80', '90–100', 'fast ice']

    mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)
    # mapper.set_array([-1, 8])

    cbar = fig.colorbar(mapper,
                        ax = ax,
                        spacing = 'uniform',
                        orientation = 'vertical',
                        pad = .01,
                        drawedges = True,
                        shrink = 0.7
    )

    cbar.set_label(label = 'WMO sea ice concentration intervals [%]')#, size = 16)
    cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)
    cbar.outline.set_linewidth(2)
    cbar.dividers.set_linewidth(2)
    cbar.outline.set_edgecolor('black')
    cbar.dividers.set_edgecolor('black')

    fig.tight_layout()
    plt.savefig('study_area.pdf', dpi = 300)



if __name__ == "__main__":
    main()
