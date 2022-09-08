import h5py
import glob
import os

import numpy as np

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature

def main():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/2021/06/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/figures/"

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = sorted(glob.glob(f"{path}*.hdf5"))[5]
    
    f = h5py.File(h5file, 'r')

    lat = f['lat'][:]
    lon = f['lon'][:]
    t2m = f['day1']['t2m'][:]
    sic = f['sic'][:]
    lsmask = f['lsmask'][:]
    oobmask = f['oobmask'][:]
    xwind = f['day1']['xwind'][:]
    sic_onehot = f['sic_target']

    """
    print(lat.shape)
    print(lon.shape)
    print(sic.shape)
    print(t2m.shape)
    print(lsmask.shape)
    print(oobmask.shape)
    print(xwind.shape)
    """

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=map_proj)
    ax.set_extent([-20, 45, 60, 90], crs=data_proj)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
    # ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')

    # ax.pcolormesh(lon, lat, t2m, transform = data_proj, zorder=3, alpha=.4)
    # ax.pcolormesh(lon, lat, xwind, transform = data_proj, zorder=3, alpha=.4)
    # ax.pcolormesh(lon, lat, sic, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    ax.pcolormesh(lon, lat, sic_onehot, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    # ax.pcolormesh(lon, lat, oobmask, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    ax.pcolormesh(lon, lat, np.ma.masked_array(lsmask, lsmask < 1), transform=data_proj, zorder=3, cmap='winter')

    plt.savefig(f"{path_figures}visual_validate_20210608.png")




if __name__ == "__main__":
    main()