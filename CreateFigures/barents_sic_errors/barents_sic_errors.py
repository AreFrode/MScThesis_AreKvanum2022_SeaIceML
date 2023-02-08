import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast")

import LambertLabels
import pyproj
import cmocean

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from netCDF4 import Dataset
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy import geodesic as cgeodesic
from shapely import geometry as sgeometry
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

def main():
    PATH_BARENTS = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/eps/barents_eps_20220103T00Z.nc"


    with Dataset(PATH_BARENTS, 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        sic = nc.variables['ice_concentration'][:]

    map_proj = ccrs.LambertConformal(central_latitude = 77.5, central_longitude = -25, standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    
    x0,y0 = PRJ(-10, 80)
    x1,y1 = PRJ(lon[-100,-100], lat[-100,-100])

    data_proj = ccrs.PlateCarree()

    LAND_highres = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        facecolor=cfeature.COLORS['land'],
                                        edgecolor='black',
                                        linewidth=.7,
                                        zorder = 2)

    xticks = [-10,  10,  30,  50, 70,80,90,100,110,120]

    yticks = [60,65,70, 75, 80, 85,90]                                

    fig = plt.figure(figsize = (12,8))
    ax = plt.axes(projection = map_proj)
    ax.set_title(f"24h mean Barents-2.5 forecast bulletin date 20220103T00Z member 0", fontsize = 20)

    cbar = ax.pcolormesh(lon, lat, np.mean(sic[:24, 0], axis=0), transform = data_proj, cmap = cmocean.cm.ice)

    circle_lat = 83.7
    circle_lon = 22.8
    circle_points = cgeodesic.Geodesic().circle(lon = circle_lon, lat = circle_lat, radius = 1.5e5, n_samples = 1000, endpoint = False)
    geom = sgeometry.Polygon(circle_points)
    
    ax.add_geometries((geom,), crs = data_proj, facecolor = 'none', edgecolor = 'red', linewidth = 2.5, zorder = 3)

    # fig.colorbar(cbar)
    ax.add_feature(LAND_highres)

    ax.set_xlim(x0,x1)
    ax.set_ylim(y0,y1)

    fig.canvas.draw()
    ax.gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgray')
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax, xticks)
    LambertLabels.lambert_yticks(ax, yticks)

    plt.savefig('barents_artefacts.png')

if __name__ == "__main__":
    main()
