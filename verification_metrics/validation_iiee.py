import h5py
import glob
import os

import numpy as np

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from shapely.errors import ShapelyDeprecationWarning

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/2021/01/iiee_20210105.hdf5"
    # path_predictions = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/2021/01/SIC_SimpleUNET_20210105T15Z.hdf5"
    path_extras = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/2021/01/PreparedSample_20210104.hdf5"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/figures/"

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()
    
    with h5py.File(path_data, 'r') as f:
        a_plus = np.where(f['a_plus'][:] == 1, 1, 0)
        a_minus = np.where(f['a_minus'][:] == 1, 1, 0)
        ocean = np.where(f['ocean'][:] == 1, 1, 0)
        ice = np.where(f['ice'][:] == 1, 1, 0)

    with h5py.File(path_extras, 'r') as f:
    #     lat = f['lat'][451::2, :1792:2]
    #     lon = f['lon'][451::2, :1792:2]
        lsmask = f['lsmask'][451::2, :1792:2]

    
    yyyymmdd = path_data[-13:-5]
    year = yyyymmdd[:4]
    month = yyyymmdd[4:6]

    save_location = f"{path_figures}{year}/{month}/"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes()
    ax.set_title(yyyymmdd, fontsize=30)
    # ax.set_extent([-20, 45, 60, 90], crs=data_proj)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')


    ax.pcolormesh(np.ma.masked_array(a_plus, a_plus < 1), zorder=3, cmap='gray')
    ax.pcolormesh(np.ma.masked_array(a_minus, a_minus < 1), zorder=3, cmap=plt.colormaps['summer'])
    ax.pcolormesh(np.ma.masked_array(ocean, ocean < 1), zorder=3, cmap=plt.colormaps['Blues_r'])
    ax.pcolormesh(np.ma.masked_array(ice, ice < 1), zorder=3, cmap=plt.colormaps['cool'])
    # ax.pcolormesh(lon, lat, a_plus == 1, zorder=4, cmap='gray')
    ax.pcolormesh(np.ma.masked_array(lsmask, lsmask < 1), zorder=4, cmap='autumn', alpha = 0.2)

    plt.savefig(f"{save_location}{yyyymmdd}_iiee.png")

    plt.close(fig)

    exit()


if __name__ == "__main__":
    main()