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
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/testing_data/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/figures/testing_data/"

    data_2019 = np.array(sorted(glob.glob(f"{path}2019/**/*.hdf5", recursive = True)))
    data_2020 = np.array(sorted(glob.glob(f"{path}2020/**/*.hdf5", recursive = True)))
    data_2021 = np.array(sorted(glob.glob(f"{path}2021/**/*.hdf5", recursive = True)))

    data = [data_2019, data_2020, data_2021]
    
    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file_constants = data[0][0]
    
    f = h5py.File(h5file_constants  , 'r')

    lat = f['lat'][:]
    lon = f['lon'][:]
    lsmask = f['lsmask'][:]

    left_lat = lat[450, 0]
    left_lon = lon[450, 0]
    right_lat = lat[450, -1]
    right_lon = lon[450, -1]

    bottom_lat = lat[0, 1840]
    bottom_lon = lon[0, 1840]
    top_lat = lat[-1, 1840]
    top_lon = lon[-1, 1840]

    left_lon_t, left_lat_t = map_proj.transform_point(left_lon, left_lat, data_proj)
    right_lon_t, right_lat_t = map_proj.transform_point(right_lon, right_lat, data_proj)

    bottom_lon_t, bottom_lat_t = map_proj.transform_point(bottom_lon, bottom_lat, data_proj)
    top_lon_t, top_lat_t = map_proj.transform_point(top_lon, top_lat, data_proj)

    for date in data[0]:
        yyyymmdd = date[-13:-5]
        print(f"{yyyymmdd}", end='\r')
        year = yyyymmdd[:4]
        month = yyyymmdd[4:6]

        save_location = f"{path_figures}{year}/{month}/"
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        f_current = h5py.File(date, 'r')
        sic_onehot = f_current['sic_target']
        sic_onehot = f_current['sic']

        # print(np.unique(sic_onehot))

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_title(date[-13:-5], fontsize=30)
        ax.set_extent([-20, 45, 60, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

        cbar = ax.pcolormesh(lon, lat, sic_onehot, transform=data_proj, zorder=2, cmap=plt.colormaps['cividis'])
        ax.pcolormesh(lon, lat, np.ma.masked_array(lsmask, lsmask < 1), transform=data_proj, zorder=3, cmap='autumn')
        ax.plot([left_lon_t, right_lon_t], [left_lat_t, right_lat_t], 'k--', transform=map_proj, zorder=4)
        ax.plot([bottom_lon_t, top_lon_t], [bottom_lat_t, top_lat_t], 'k--', transform=map_proj, zorder=4)

        plt.colorbar(cbar)

        plt.savefig(f"{save_location}{yyyymmdd}.png")

        f_current.close()
        plt.close(fig)

        exit()


    h5file_constants.close()

if __name__ == "__main__":
    main()