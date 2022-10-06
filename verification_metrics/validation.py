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
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/2021/01/ice_edge_20210105.hdf5"
    path_predictions = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/weights_20091742/2021/01/SIC_SimpleUNET_20210105T15Z.hdf5"
    path_extras = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/2021/01/PreparedSample_20210105.hdf5"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/figures/"

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()
    
    with h5py.File(path_data, 'r') as f:
        ice_edge = f['ice_edge_target'][:]

    
    with h5py.File(path_extras, 'r') as f:
        lat = f['lat'][451::2, :1792:2]
        lon = f['lon'][451::2, :1792:2]
        lsmask = f['lsmask'][451::2, :1792:2]
        sic = f['sic_target'][451::2, :1792:2]

    with h5py.File(path_predictions, 'r') as f:
        sic_pred = f['y_pred'][0]
 
    yyyymmdd = path_data[-13:-5]
    year = yyyymmdd[:4]
    month = yyyymmdd[4:6]

    save_location = f"{path_figures}{year}/{month}/"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes()
    # ax = plt.axes(projection=map_proj)
    ax.set_title(yyyymmdd, fontsize=30)
    # ax.set_extent([-20, 45, 60, 90], crs=data_proj)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

    cbar = ax.pcolormesh(sic, zorder=2, cmap=plt.colormaps['cividis'])
    ax.pcolormesh(np.ma.masked_array(ice_edge, ice_edge < 1), zorder=3, cmap=plt.colormaps['spring'])
    # plt.pcolormesh(sic, zorder=2, cmap=plt.colormaps['cividis'])
    # plt.pcolormesh(np.ma.masked_array(ice_edge, ice_edge < 1), zorder=3, cmap=plt.colormaps['spring'])
    ax.pcolormesh(np.ma.masked_array(lsmask, lsmask < 1), zorder=4, cmap='autumn')

    plt.colorbar(cbar)

    plt.savefig(f"{save_location}{yyyymmdd}_target.png")

    plt.close(fig)

    exit()


if __name__ == "__main__":
    main()