import h5py
import glob
import os

import numpy as np

from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature


def runstuff():
    path = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/2019/01/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/figures/"

    if not os.path.exists(path_figures):
        os.mkdir(path_figures)

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    h5file = glob.glob(f"{path}*.hdf5")[0]
    
    f = h5py.File(h5file, 'r')

    lat = f['lat']
    lon = f['lon']
    t2m = f['day1']['t2m']
    sic = f['sic']
    oob = f['oobmask']
    lsmask = f['lsmask']
    sst = f['sst']


    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=map_proj)
    ax.set_extent([-20, 45, 60, 90], crs=data_proj)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
    # ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')

    # ax.pcolormesh(lon, lat, t2m, transform = data_proj, zorder=3, alpha=.7)
    # ax.pcolormesh(lon, lat, sic, transform=data_proj, zorder=2, cmap=plt.colormaps['PiYG'])
    ax.pcolormesh(lon, lat, lsmask, transform=data_proj, zorder=2, cmap=plt.colormaps['cividis'], alpha=1.)

    plt.savefig(f"{path_figures}visual_validate.png")
    

if __name__ == "__main__":
    runstuff()