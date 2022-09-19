import h5py
import glob
import os

import numpy as np

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature

def main():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/2020/09/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/figures/"

    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[15]
    date = h5file[-13:-5]
    
    f = h5py.File(h5file, 'r')

    lat = f['lat'][:]
    lon = f['lon'][:]
    t2m = f['day1']['t2m'][:]
    sic = f['sic'][:]
    lsmask = f['lsmask'][:]
    xwind = f['day1']['xwind'][:]
    sic_onehot = f['sic_target']

    left_lat = lat[451, 0]
    left_lon = lon[451, 0]
    right_lat = lat[451, -1]
    right_lon = lon[451, -1]

    bottom_lat = lat[0, 1792]
    bottom_lon = lon[0, 1792]
    top_lat = lat[-1, 1792]
    top_lon = lon[-1, 1792]

    left_lon_t, left_lat_t = map_proj.transform_point(left_lon, left_lat, data_proj)
    right_lon_t, right_lat_t = map_proj.transform_point(right_lon, right_lat, data_proj)

    bottom_lon_t, bottom_lat_t = map_proj.transform_point(bottom_lon, bottom_lat, data_proj)
    top_lon_t, top_lat_t = map_proj.transform_point(top_lon, top_lat, data_proj)

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=map_proj)
    ax.set_title(date, fontsize=30)
    ax.set_extent([-20, 45, 60, 90], crs=data_proj)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

    cbar = ax.pcolormesh(lon, lat, sic_onehot, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    ax.pcolormesh(lon, lat, np.ma.masked_array(lsmask, lsmask < 1), transform=data_proj, zorder=3, cmap='winter')
    ax.plot([left_lon_t, right_lon_t], [left_lat_t, right_lat_t], 'k--', transform=map_proj, zorder=4)
    ax.plot([bottom_lon_t, top_lon_t], [bottom_lat_t, top_lat_t], 'k--', transform=map_proj, zorder=4)

    plt.colorbar(cbar)

    plt.savefig(f"{path_figures}visual_validate_{date}_sictarget.png")




if __name__ == "__main__":
    main()