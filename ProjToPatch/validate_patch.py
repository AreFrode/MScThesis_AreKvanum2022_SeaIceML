# This script will plot some pathces using cartopy to visually inspect that everything is working as it should

import glob
import numpy as np

from matplotlib import patches as mpatches
from calendar import monthrange
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt

from patch_bins import get_valid_patches, sliding_window_no_time, sliding_window_from_idx, determine_meanSIC, find_nearest


def icechart_patch_from_idx(ic, x_idx, y_idx, stride=250):
    """Modified read patch from icechhart based on patch/patch_bins.py

    Args:
        ic (ndarray): Ice concentreation as a 2d field covering the AA grid
        x_idx (List[int]): List containing valid x-coordinate values
        y_idx (List[int]): List containing valid y-coordinate values
        stride (int, optional): Uniform size of patch. Defaults to 250.

    Returns:
        ndarray(List(ndarray)): Patches of SIC, with third dimension reprsenting patch index
    """
    outputs = []
    for y,x in zip(y_idx, x_idx):
        current_chart = ic[..., y:y+stride,x:x+stride]
        current_chart = current_chart[0].filled(np.nan)

        indices = distance_transform_edt(np.isnan(current_chart), return_distances=False, return_indices=True)
        current_chart = current_chart[tuple(indices)]

        outputs.append(current_chart)
        
    return np.array(outputs, dtype=np.float32)

def main():
    # Setup data
    path_IceChart = "/lustre/storeB/project/copernicus/sea_ice/SIW-METNO-ARC-SEAICE_HR-OBS/"
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/"
    path_figures = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/figures/"

    map_proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()

    paths = []
    for year in range(2019, 2022):
        for month in range(1, 13):
            p = f"{path_data}{year}/{month:02d}/"
            paths.append(p)


    # path_data_task = paths[$SGE_TASK_ID - 1]
    path_data_task = paths[14]
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
        ic_path = glob.glob(f"{path_IceChart}{year_task}/{month_task}/ice_conc_svalbard_{yyyymmdd}1500.nc")[0]

        nc = Dataset(arome_path, 'r')
        x = nc.variables['xc']
        y = nc.variables['yc']
        temp = nc.variables['T2M']
        sst = nc.variables['SST']
        xwind = nc.variables['X_wind_10m']
        ywind = nc.variables['Y_wind_10m']

        nc_IC = Dataset(ic_path, 'r')

        xc = nc_IC['xc'][:]
        yc = nc_IC['yc'][:]

        lat = nc_IC['lat'][:]
        lon = nc_IC['lon'][:]

        xmin = find_nearest(xc, x[:].min())
        xmax = find_nearest(xc, x[:].max())
        ymin = find_nearest(yc, y[:].min())
        ymax = find_nearest(yc, y[:].max())

        sic = nc_IC['ice_concentration'][..., ymin:ymax, xmin:xmax]

        SIC_thresholds = determine_meanSIC(sic)

        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection=map_proj)
        ax.set_extent([-20, 45, 60, 90], crs=data_proj)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')

        for key in SIC_thresholds.keys():
            x_idx = np.array(SIC_thresholds[key]['x'])
            y_idx = np.array(SIC_thresholds[key]['y'])
            x_idx, y_idx = get_valid_patches(temp, x_idx, y_idx)
            lats = sliding_window_no_time(lat, x_idx, y_idx)
            lons = sliding_window_no_time(lon, x_idx, y_idx)
            icechart = icechart_patch_from_idx(sic, x_idx, y_idx)

            for idx, i in enumerate(range(len(lons[:,0,0]))):
                ax.pcolormesh(lons[i], lats[i], icechart[i], transform=data_proj, vmin=0, vmax=100)
                ax.text(lons[i,0,0], lats[i,0,0], f"{key}:{idx}", transform=data_proj, size='large')

            
        plt.savefig(f"{path_figures}mosaic_test.png")

        nc.close()
        nc_IC.close()



if __name__ == "__main__":
    main()