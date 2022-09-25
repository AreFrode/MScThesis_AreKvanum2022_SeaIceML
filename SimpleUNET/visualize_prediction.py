import h5py
import glob
import os

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from shapely.errors import ShapelyDeprecationWarning

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/2021/01/"
    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/figures/"
    
    data_2021 = np.array(sorted(glob.glob(f"{path_pred}2021/**/*.hdf5", recursive=True)))

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[0]

    f = h5py.File(h5file, 'r')
    lat = f['lat'][451::2, :1792:2]
    lon = f['lon'][451::2, :1792:2]
    lsmask = f['lsmask'][451::2, :1792:2]

    for date in data_2021:
        yyyymmdd = date[-17:-9]
        print(f"{yyyymmdd}")
        year = yyyymmdd[:4]
        month = yyyymmdd[4:6]

        save_location = f"{path_figures}{year}/{month}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        f_model = h5py.File(f"{path_pred}{year}/{month}/SIC_SimpleUNET_{yyyymmdd}T15Z.hdf5", 'r')

        y_pred = f_model['y_pred'][0]

        cmap = plt.get_cmap('cividis', 7)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_title(yyyymmdd, fontsize = 30)
        ax.set_extent([-18, 45, 65, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        # ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')

        # ax.pcolormesh(lon, lat, t2m, transform = data_proj, zorder=3, alpha=1.)
        # ax.pcolormesh(lon, lat, xwind, transform = data_proj, zorder=2, alpha=1.)
        # ax.pcolormesh(lon, lat, sic, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
        # ax.pcolormesh(lon, lat, y, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
        cbar = ax.pcolormesh(lon, lat, y_pred, transform=data_proj, zorder=2, cmap=cmap)
        ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap='autumn')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, map_projection=ccrs.PlateCarree())
        plt.colorbar(cbar, cax = cax)

        plt.savefig(f"{save_location}{yyyymmdd}.png")

        f_model.close()
        plt.close(fig)

    f.close()


if __name__ == "__main__":
    main()