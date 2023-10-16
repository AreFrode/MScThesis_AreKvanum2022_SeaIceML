import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts')
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")

import h5py
import glob
import os
import WMOcolors
import pyproj

import numpy as np
import seaborn as sns
import matplotlib as mpl

from matplotlib import pyplot as plt, colors as mcolors
from netCDF4 import Dataset
from cartopy import crs as ccrs

from datetime import datetime, timedelta
from helper_functions import read_config_from_csv
from verification_metrics import IIEE_alt
from loadClimatologicalIceEdge import load_climatological_ice_edge


def main():
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/"
    
    data_2022 = np.array(sorted(glob.glob(f"{path_pred}{weights}/2022/**/*.hdf5")))

    h5file = sorted(glob.glob(f"{path}/01/*.hdf5"))[0]

    with h5py.File(h5file, 'r') as f:
        lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
        lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    inspection_months = ['03', '06', '09', '12']
    inspection_months_names = ['March', 'June', 'September', 'December']
    inspection_dates = [24, 59, 98, 136]
    inspection_names = ['right-left', 'top-bottom', 'bottom-top', 'uniform-max', 'uniform-min', 'left-right', 'both-neg', 'both-pos', 'no-wind', 'only-xneg', 'only-xpos', 'only-yneg', 'only-ypos']

    conc = '15%'
    lead_time = 2
    climatological_ice_edge = load_climatological_ice_edge(2022, conc, lead_time)
    side_length = 1
    threshold = 2

    sns.set_theme()

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()
    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])
    land_cmap = WMOcolors.cm.land()

    norm = mcolors.Normalize(vmin = -6, vmax = 6)
    
    cmap = plt.cm.seismic
    cmaplist = [cmap(i) for i in range(cmap.N)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # discrete_cmap = 
    levels = np.linspace(-6, 6, 14)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    cbar_labels = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5', '+6']

    for date, month, month_name in zip(inspection_dates, inspection_months, inspection_months_names):
        NIIEE = []
        yyyymmdd = data_2022[date][-17:-9]
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = config['lead_time'])).strftime('%Y%m%d')
        print(yyyymmdd)

        clim_ice_edge = climatological_ice_edge[conc].loc[yyyymmdd]

        with h5py.File(data_2022[date], 'r') as f_baseline:
            pred_baseline = f_baseline['y_pred'][0]

        with h5py.File(f"{path}{month}/PreparedSample_v{yyyymmdd_valid}_b{yyyymmdd}.hdf5", 'r') as f_target:
            sic_target = f_target['sic_target'][config['lower_boundary']:, :config['rightmost_boundary']]

        iiee_baseline = IIEE_alt(pred_baseline, sic_target, lsmask, side_length = side_length, threshold = threshold)

        NIIEE.append((iiee_baseline[0].sum() + iiee_baseline[1].sum()) / clim_ice_edge)

        synthetics = sorted(glob.glob(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/SyntheticForcing/Data/weights_21021550/2022/{month}/SIC_UNET_*_{date}_*.nc"))
        
        forecasts = []

        for synth in synthetics:
            with Dataset(synth, 'r') as nc:
                forecasts.append(nc.variables['y_pred'][:])

        forecasts = np.array(forecasts)
        pred_baselines = np.repeat(pred_baseline[np.newaxis, :, :], forecasts.shape[0], axis = 0)

        biases = forecasts - pred_baselines

        for forecast in forecasts:
            iiee = IIEE_alt(forecast, sic_target, lsmask, side_length = side_length, threshold = threshold)
            NIIEE.append((iiee[0].sum() + iiee[1].sum()) / clim_ice_edge)

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9, 1.5))

        x = np.arange(len(forecasts))
        ax.plot(x, NIIEE[1:],  linestyle = '--', marker = 'o', color = '#ff7f0e')
        ax.axvline(x = 5.5, color = 'k', ls = '--')
        ax.axhline(y = NIIEE[0], xmin = 0, xmax = len(forecasts), linestyle = '--', label = 'Default Deep learning')

        ax.text(0.1, 0.55, 't2m', transform = ax.transAxes, bbox=dict(edgecolor = 'k', facecolor = '#EAEAF2'))
        ax.text(0.75, 0.55, 'winds', transform = ax.transAxes, bbox=dict(edgecolor = 'k', facecolor = '#EAEAF2'))


        labels = [('\n' if i % 2 == 1 else '') + inspection_names[i] for i in range(len(inspection_names))]

        ax.set_xticks(x, labels)
        ax.legend()

        ax.set_title(f'NIIEE for different synthetic AROME Arctic fields (10%) contour at {month_name}')
        ax.set_xlabel('Modified AROME Arctic field')
        ax.set_ylabel('NIIEE [km]')
            
        fig.savefig(f'NIIEE_{month}.pdf', bbox_inches = 'tight')

        '''
        for bias, name in zip(biases, inspection_names):

            fig = plt.figure(figsize=(13.5,13.5))
            # fig.subplots_adjust(bottom = 0.2)
            ax = plt.axes(projection=map_proj)

            ax.set_title(f'Prediction deviation with {name} at {month_name}', fontsize = 30)

            ax.pcolormesh(lon, lat, bias, transform = data_proj, norm = norm, cmap = cmap, zorder = 1)
            ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform = data_proj, zorder = 2, cmap = land_cmap)

            ax.set_xlim(x0,x1)
            ax.set_ylim(y0,y1)

            mapper = mpl.cm.ScalarMappable(cmap = cmap, norm = norm)
            cbar = fig.colorbar(mapper, ax = ax, spacing = 'uniform', location = 'bottom', orientation = 'horizontal')
            cbar.set_label(label = 'Category difference', fontsize = 30)
            cbar.set_ticks(levels[:-1] + .5, labels = cbar_labels, fontsize = 30)

            save_path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/SyntheticForcing/figures/{month}/"

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fig.savefig(f"{save_path}bias_{name}.png")
        
        '''
        plt.close('all')

    exit()
        



if __name__ == "__main__":
    main()
