import os
import glob
import numpy as np

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from cartopy import crs as ccrs
from cartopy import feature as cfeature



def main():
    # setup data
    PATH_DATA = "/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/"
    PATH_AROME = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    PATH_ICECHART = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/NSIDC_regional_mask/figures/"

    if not os.path.exists(PATH_FIGURES):
        os.mkdir(PATH_FIGURES)

    # Read data
    mask_path = glob.glob(f"{PATH_DATA}sio_2016_mask.nc")
    arome_path = glob.glob(f"{PATH_AROME}2019/01/AROME_ICgrid_20190115T00Z.nc")
    icechart_path = glob.glob(f"{PATH_ICECHART}/2019/01/ice_conc_svalbard_201901151500.nc")

    nc = Dataset(mask_path[0])
    
    lats = nc['lat'][:]
    lons = nc['lon'][:]
    mask = nc['mask'][:]

    nc_arome = Dataset(arome_path[0])
    
    t2m = nc_arome['T2M'][0,...]
    arome_lats = nc_arome['lat'][:]
    arome_lons = nc_arome['lon'][:]

    nc_ic = Dataset(icechart_path[0])
    ic = nc_ic['ice_concentration'][0, :]
    ic_lat = nc_ic['lat'][:]
    ic_lon = nc_ic['lon'][:]
    
    arctic_regions = [7,8,9,15] #E.Greenland, Barents, Kara, C.Arctic

    mask = np.where(np.isin(mask, arctic_regions), mask, np.nan)

    # Plot data
    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()


    fig, ax = plt.subplots(figsize=(20,20),subplot_kw={'projection': map_proj})
    ax.set_extent([-180, 180, 60, 90], crs=data_proj)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
    ax.add_feature(cfeature.LAND, zorder=4, edgecolor='black', facecolor='none')

    ax.pcolormesh(lons, lats, mask, transform=data_proj, cmap='tab10')
    cbar = ax.pcolormesh(arome_lons, arome_lats, t2m, transform=data_proj, alpha=.4, zorder=3)
    fig.colorbar(cbar)
    ax.pcolormesh(ic_lon, ic_lat, ic, transform=data_proj, cmap='cividis', alpha=.4, zorder=2)


    plt.savefig(f"{PATH_FIGURES}visualize_mask.png")


    nc.close()
    nc_arome.close()
    nc_ic.close()


if __name__ == "__main__":
    main()