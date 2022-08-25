import glob
import os
import numpy as np

from matplotlib import patches as mpatches
from calendar import monthrange
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from netCDF4 import Dataset
from get_boundaries import find_nearest


def main():
    # Setup data
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/testingdata/"
    path_sic = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/figures/"

    if not os.path.exists(path_figures):
        os.mkdir(path_figures)

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    paths = []
    paths_sic = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{path_data}{year}/{month:02d}/"
            p_sic = f"{path_sic}{year}/{month:02d}/"
            paths.append(p)
            paths_sic.append(p_sic)


    # path_data_task = paths[$SGE_TASK_ID - 1]
    path_data_task = paths[0]
    path_sic_task = paths_sic[0]

    print(f"path_data_task = {path_data_task}")
    year_task = path_data_task[len(path_data) : len(path_data) + 4]
    print(f"year_task = {year_task}")
    month_task = path_data_task[len(path_data) + 5 : len(path_data) + 7]
    print(f"month_task = {month_task}")
    nb_days_task = monthrange(int(year_task), int(month_task))[1]
    print(f"nb_days_task = {nb_days_task}")
    

    for dd in range(2, 3):
        yyyymmdd = f"{year_task}{month_task}{dd:02d}"
        print(f"{yyyymmdd}")

        arome_path = glob.glob(f"{path_data_task}AROME_ICgrid_{yyyymmdd}T00Z.nc")[0]
        sic_path = glob.glob(f"{path_sic_task}ice_conc_svalbard_{yyyymmdd}1500.nc")[0]


        nc = Dataset(arome_path, 'r')
        x_min = nc.variables['xc'][:].min()
        x_max = nc.variables['xc'][:].max()
        y_min = nc.variables['yc'][:].min()
        y_max = nc.variables['yc'][:].max()
        lat = nc.variables['lat']
        lon = nc.variables['lon']
    
        
        nc_ic = Dataset(sic_path, 'r')
        xc = nc_ic.variables['xc']
        yc = nc_ic.variables['yc']

        xc_min = find_nearest(xc[:], x_min)[0]
        xc_max = find_nearest(xc[:], x_max)[0]
        yc_min = find_nearest(yc[:], y_min)[0]
        yc_max = find_nearest(yc[:], y_max)[0]

        lonc = nc_ic.variables['lon'][yc_min:yc_max, xc_min:xc_max]
        latc = nc_ic.variables['lat'][yc_min:yc_max, xc_min:xc_max]
        sic = nc_ic.variables['ice_concentration'][0, yc_min:yc_max, xc_min:xc_max]
        
        fields = ['T2M', 'LSMASK', 'X_wind_10m', 'Y_wind_10m']

        oob_mask = np.where(np.isnan(nc.variables[fields[0]][0,:]), 1, np.nan) # out of bounds mask

        for field in fields:
            print(f"Plotting {field}")
            fig = plt.figure(figsize=(20,20))
            ax = plt.axes(projection=map_proj)
            ax.set_extent([-20, 45, 60, 90], crs=data_proj)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
            ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')
            ax.pcolormesh(lon[:], lat[:], nc.variables[field][0,:], transform=data_proj, zorder=4, alpha=.7)
            
            ax.pcolormesh(lonc, latc, sic, transform=data_proj, zorder=3, cmap=plt.colormaps['PiYG'])
            ax.pcolormesh(lon[:], lat[:], oob_mask, transform=data_proj, zorder=4, alpha = .7, cmap=plt.colormaps['cividis'])

            plt.savefig(f"{path_figures}visualize_{yyyymmdd}_{field}.png")

        SST = nc.variables['SST'][0,:]
        SST = np.where(SST > 0, SST, np.nan)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_extent([-20, 45, 60, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black', facecolor='none')
        ax.pcolormesh(lon[:], lat[:], SST, transform=data_proj, zorder=2)

        plt.savefig(f"{path_figures}visualize_SST.png")

        nc.close()
        nc_ic.close()


if __name__ == "__main__":
    main()