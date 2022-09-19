import h5py
import glob
import os

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature

def main():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/2020/06/"
    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/predictions/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/figures/"

    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[3]

    fname = "16091629"

    f = h5py.File(h5file, 'r')
    # f_model = h5py.File(f"{path_pred}pred_runmodel{fname}.hdf5", 'r')
    f_model = h5py.File(f"{path_pred}pred_0_{fname}.hdf5", 'r')

    lat = f['lat'][451::2, :1792:2]
    lon = f['lon'][451::2, :1792:2]
    lsmask = f['lsmask'][451::2, :1792:2]

    # y = f_model['y'][0]
    y_pred = f_model['y_pred'][0]
    date = f_model['date'][()]

    cmap = plt.get_cmap('PiYG', 6)

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=map_proj)
    ax.set_title(date.decode('UTF-8'), fontsize = 30)
    ax.set_extent([-18, 45, 65, 90], crs=data_proj)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
    # ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')

    # ax.pcolormesh(lon, lat, t2m, transform = data_proj, zorder=3, alpha=1.)
    # ax.pcolormesh(lon, lat, xwind, transform = data_proj, zorder=2, alpha=1.)
    # ax.pcolormesh(lon, lat, sic, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    # ax.pcolormesh(lon, lat, y, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    cbar = ax.pcolormesh(lon, lat, y_pred, transform=data_proj, zorder=2, cmap=cmap)
    ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap='winter')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, map_projection=ccrs.PlateCarree())
    plt.colorbar(cbar, cax = cax)
    

    plt.savefig(f"{path_figures}visual_prediction_0_{fname}.png")

    f.close()
    f_model.close()


if __name__ == "__main__":
    main()