import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")


import h5py
import glob
import os
import LambertLabels
import pyproj
import WMOcolors

import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt, transforms as mtransforms
from cartopy import crs as ccrs
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from netCDF4 import Dataset
from tqdm import tqdm
from computeMetrics import load_barents, load_ml, load_nextsim, load_osisaf, load_persistence


from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    # sns.despine()

    sns.set_theme('talk')
    sns.set(font_scale = 2)
    lead_time = sys.argv[1]
    grid = 'nextsim'
    weights = sys.argv[2]

    # Define paths
    csv_PATH_NEXTSIM = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/nextsim.csv"
    csv_PATH_OSISAF = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/osisaf.csv"
    csv_PATH_ML = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/{weights}.csv"
    csv_PATH_BARENTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/barents.csv"
    csv_PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/persistence.csv"

    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/figures/thesis_figs/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)
    
    # Read available statistics files

    # files = (PATH_NEXTSIM, PATH_OSISAF, PATH_ML, PATH_BARENTS, PATH_PERSISTENCE)
    # files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)
    # files = [PATH_NEXTSIM, PATH_ML, PATH_BARENTS]

    files = [csv_PATH_NEXTSIM, csv_PATH_PERSISTENCE, csv_PATH_ML, csv_PATH_OSISAF,  csv_PATH_BARENTS]

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()


    ice_cmap = mpl.colormaps['RdBu']
    ice_colors = ice_cmap(np.linspace(0, 1, 13))
    ice_cmap = colors.ListedColormap(ice_colors)

    ice_levels = np.linspace(-6, 7, 14, dtype = 'int')
    ice_norm = colors.BoundaryNorm(ice_levels, ice_cmap.N)

    ice_ticks = [f'{i}' for i in range(-6,7)]

    land_cmap = WMOcolors.cm.land()

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/2022/01/"
    
    with h5py.File(sorted(glob.glob(f"{path}*.hdf5"))[0], 'r') as f:
        lat_vis = f['lat'][578:, :1792]
        lon_vis = f['lon'][578:, :1792]


    commons = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim_commons.nc"

    with Dataset(commons, 'r') as nc_common:
        lsmask = nc_common['lsmask'][:]
        lat = nc_common['lat'][:]
        lon = nc_common['lon'][:]

    x0,y0 = PRJ(lon_vis[0,0], lat_vis[0,0])
    x1,y1 = PRJ(lon_vis[-1,-1], lat_vis[-1,-1])

    dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array


    ml_anom = []
    osi_anom = []
    nextsim_anom = []
    barents_anom = []
    pers_anom = []
    dates_datetime = []

    barents_test = []

    for date in tqdm(dates):
        date_datetime = datetime.strptime(date, '%Y-%m-%d')
        date_bulletin = date_datetime.strftime('%Y%m%d')
        date_bulletin_physical = (date_datetime + timedelta(days = 1)).strftime('%Y%m%d')
        date_valid = (date_datetime + timedelta(days = 2)).strftime('%Y%m%d')

        
        sic_ml, sic_target, _ = load_ml(date_bulletin, int(lead_time), grid, PATH_TARGETS, weights)
        
        sic_osi, _, _ = load_osisaf(date_bulletin, int(lead_time), grid, PATH_TARGETS, 2)
        sic_nextsim, _, _ = load_nextsim(date_bulletin_physical, int(lead_time), grid, PATH_TARGETS, None)
        sic_barents, _, _ = load_barents(date_bulletin_physical, int(lead_time), grid, PATH_TARGETS, None)
        sic_pers, _, _ = load_persistence(date_bulletin, int(lead_time), grid, PATH_TARGETS, None)

        # sic_barents = np.where(lsmask == 1, -10, sic_barents)
        sic_target = np.where(sic_ml == -1, -1, sic_target)

        ml_anom.append(sic_ml - sic_target)
        osi_anom.append(sic_osi - sic_target)
        nextsim_anom.append(sic_nextsim - sic_target)
        barents_anom.append(sic_barents - sic_target)
        pers_anom.append(sic_pers - sic_target)

        barents_test.append(sic_barents)

        dates_datetime.append(np.datetime64(date_datetime + timedelta(days = 2)).astype('datetime64[M]'))


    ml_anom = np.array(ml_anom)
    osi_anom = np.array(osi_anom) 
    nextsim_anom = np.array(nextsim_anom) 
    barents_anom = np.array(barents_anom) 
    pers_anom = np.array(pers_anom)

    dates_datetime = np.array(dates_datetime)
    
    DJF = np.where(np.logical_or(np.logical_or(dates_datetime == np.datetime64('2022-01'), dates_datetime == np.datetime64('2022-02')), dates_datetime == np.datetime64('2022-12')))

    MAM = np.where(np.logical_or(np.logical_or(dates_datetime == np.datetime64('2022-03'), dates_datetime == np.datetime64('2022-04')), dates_datetime == np.datetime64('2022-05')))

    JJA = np.where(np.logical_or(np.logical_or(dates_datetime == np.datetime64('2022-06'), dates_datetime == np.datetime64('2022-07')), dates_datetime == np.datetime64('2022-08')))

    SON = np.where(np.logical_or(np.logical_or(dates_datetime == np.datetime64('2022-09'), dates_datetime == np.datetime64('2022-10')), dates_datetime == np.datetime64('2022-11')))

    seasons = [DJF, MAM, JJA, SON]
    products = [nextsim_anom, pers_anom, ml_anom, osi_anom, barents_anom]
    
    fig, ax = plt.subplots(nrows = 5, ncols = 4, figsize = (15, 15), subplot_kw = {'projection' : map_proj})

    normalize = colors.Normalize(vmin = -6, vmax = 6)

    for i, prod in zip(range(4), products[:-1]):
        for j, season in zip(range(4), seasons):
            ax[i, j].pcolormesh(lon, lat, np.where(lsmask == 1, 0, prod[season].mean(axis = 0)), cmap = 'seismic', zorder=1, rasterized = True, norm = normalize, transform = data_proj)
            # ax[i, j].pcolormesh(lon, lat, prod[season].mean(axis = 0), cmap = 'seismic', zorder=1, rasterized = True, norm = normalize, transform = data_proj)
            # ax[i, j].pcolormesh(lon, lat, np.ma.masked_less(np.where(sic_ml == -1, 0, lsmask) , 1), zorder=2, cmap=land_cmap, rasterized = True, transform = data_proj)
            ax[i, j].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), zorder=2, cmap=land_cmap, rasterized = True, transform = data_proj)
            
            ax[i, j].set_xlim(x0,x1)
            ax[i, j].set_ylim(y0,y1)


    fig.delaxes(ax[4,0])
    fig.delaxes(ax[4,1])


    # print(np.unique(ml_anom[JJA].mean(axis = 0)))
    # print(np.unique(nextsim_anom[JJA].mean(axis = 0)))


    # print(np.unique(barents_anom[JJA].mean(axis = 0)))

    for i, season in zip(range(2, 4), seasons[2:]):
        ax[4,i].pcolormesh(lon, lat, barents_anom[season].mean(axis = 0), transform = data_proj, cmap = 'seismic', norm = normalize)
        ax[4,i].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), zorder=2, cmap=land_cmap, rasterized = True, transform = data_proj)
        ax[4,i].set_xlim(x0,x1)
        ax[4,i].set_ylim(y0, y1)

    ax[0,0].set_title('DJF')
    ax[0,1].set_title('MAM')
    ax[0,2].set_title('JJA')
    ax[0,3].set_title('SON')

    ax[0,0].text(-0.07, 0.55, 'neXtSIM', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax[0,0].transAxes)
    ax[1,0].text(-0.07, 0.55, 'Persistence', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax[1,0].transAxes)
    ax[2,0].text(-0.07, 0.55, 'Deep learning', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax[2,0].transAxes)
    ax[3,0].text(-0.07, 0.55, 'OSI SAF trend', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax[3,0].transAxes)
    ax[4,2].text(-0.07, 0.55, 'Barents-2.5', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax[4,2].transAxes)

    
    mapper = mpl.cm.ScalarMappable(cmap = 'seismic', norm = normalize)
    fig.colorbar(mapper, ax = ax[4, :2], orientation = 'horizontal', label = 'Bias [mean sea ice category]', anchor = (0.5, 0.5))

    fig.suptitle('Seasonal distribution of spatial biases')


    fig.savefig(f"{PATH_FIGURES}anomalies.pdf", dpi=300)


if __name__ == "__main__":
    main()
