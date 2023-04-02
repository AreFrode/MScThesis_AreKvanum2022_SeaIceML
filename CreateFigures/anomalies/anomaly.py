import sys
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")

import h5py
import glob
import LambertLabels
import pyproj

import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
import WMOcolors

from matplotlib import pyplot as plt, transforms as mtransforms
from cartopy import crs as ccrs
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from netCDF4 import Dataset
from createHDF import onehot_encode_sic
from scipy.interpolate import NearestNDInterpolator
from verification_metrics import find_ice_edge, IIEE


from datetime import datetime
from helper_functions import read_config_from_csv

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    path_raw = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/01/"
    # path_figure = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/lustre_poster_extra/"
    path_figure = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/anomalies/"


    lower_boundary = 578
    rightmost_boundary = 1792

    year = '2022'
    month = '03'
    bday = '23'
    vday = '25'

    vdate = datetime.strptime(f"{year}{month}{vday}", '%Y%m%d')

    data_ml = glob.glob(f"{path_pred}{weights}/{year}/{month}/SIC_UNET_v{year}{month}{vday}_b{year}{month}{bday}T15Z.hdf5")[0]

    data_ic = glob.glob(f"{path_raw}{year}/{month}/ICECHART_1kmAromeGrid_{year}{month}{vday}T1500Z.nc")[0]


    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    yticks = [60,65,70, 75, 80, 85,90]

    ice_cmap = mpl.colormaps['RdBu']
    ice_colors = ice_cmap(np.linspace(0, 1, 13))
    ice_cmap = colors.ListedColormap(ice_colors)

    ice_levels = np.linspace(-6, 7, 14, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    ice_ticks = [f'{i}' for i in range(-6,7)]

    land_cmap = WMOcolors.cm.land()

    sns.set_theme(context='paper')
    figsize = (10,10)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    ax = []

    ax.append(fig.add_subplot(111, projection = map_proj))

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

    with Dataset(data_ic, 'r') as ic:
        sic_ic = onehot_encode_sic(ic['sic'][lower_boundary:, :rightmost_boundary])
    
    sic_ic_interpolator = NearestNDInterpolator(mask_T, sic_ic[mask])
    sic_ic = sic_ic_interpolator(*np.indices(sic_ic.shape))

    with h5py.File(data_ml, 'r') as ml:
        sicml = ml['y_pred'][0,:,:]

    
    anomalies = sicml - sic_ic

    ax[0].pcolormesh(lon, lat, anomalies, transform=data_proj, cmap = ice_cmap, norm = ice_norm, zorder=1, rasterized = True)
    ax[0].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap=land_cmap, rasterized = True)

    ax[0].set_xlim(x0,x1)
    ax[0].set_ylim(y0,y1)

    ax[0].set_title(f"Anomaly between Deep learning and Ice Chart {vdate.strftime('%d')}th {vdate.strftime('%B')} {vdate.strftime('%Y')}")
    
    fig.canvas.draw()
    ax[0].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[0].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[0].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[0], xticks)
    LambertLabels.lambert_yticks(ax[0], yticks)

    mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)


    cbar = fig.colorbar(mapper, 
                        ax = ax[0],
                        spacing = 'uniform',
                        orientation = 'vertical'
                        # drawedges = True
    )

    cbar.set_label(label = 'Category anomaly')
    cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)


    print('saving fig')
    fig.savefig(f"{path_figure}anomalies.pdf")

if __name__ == "__main__":
    main()