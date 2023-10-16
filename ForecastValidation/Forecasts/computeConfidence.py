import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/')

import os
import pandas as pd
import glob
import h5py
import pyproj
import cmocean

import numpy as np

from helper_functions import read_config_from_csv

from matplotlib import pyplot as plt, transforms as mtransforms
from cartopy import crs as ccrs


def main():
    model_name = sys.argv[1]
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/{model_name}/"



    config = read_config_from_csv(f"{PATH_FORECAST[:-22]}configs/{model_name}.csv")

    PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/"
    
    icechart = sorted(glob.glob(f"{PATH_PERSISTENCE}2022/01/*.hdf5"))[0]

    with h5py.File(icechart, 'r') as constants:
        lat = constants['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = constants['lon'][config['lower_boundary']:, :config['rightmost_boundary']]




    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/weights_21021550"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    forecasts = sorted(glob.glob(f"{PATH_FORECAST}confidence/2022/**/*.hdf5"))



    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    djf = []
    mam = []
    jja = []
    son = []

    print(len(forecasts))
    for forecast in forecasts:
        confidences = np.empty((config['num_outputs'], config['height'], config['width']))
        with h5py.File(forecast, 'r') as f:
            for i in range(config['num_outputs']):
                confidences[i] = f[f'confidences/contour_{i}'][:]

        month = int(forecast[-40:-38])
        print(month, end='\r')
        
        if month in [1,2,12]:
            djf.append(confidences)

        elif month in [3,4,5]:
            mam.append(confidences)

        elif month in [6,7,8]:
            jja.append(confidences)

        else:
            son.append(confidences)

    labels = ['a', 'b', 'c', 'd']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    contours = ['>= 0%', '> 0%', '> 10%', '> 40%', '> 70%', '> 90%', '100%']


    djf = np.array(djf)
    mam = np.array(mam)
    jja = np.array(jja)
    son = np.array(son)

    
    for i, cont in zip(range(config['num_outputs']), contours):
        seasonal_means = np.array([djf[:,i].mean(axis=0), mam[:,i].mean(axis=0), jja[:,i].mean(axis=0), son[:,i].mean(axis=0)])
        fig, ax = plt.subplot_mosaic('''
                                     ab
                                     cd
                                     ''', subplot_kw={'projection': map_proj})
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

        for j, lab, season in zip(range(4), labels, seasons):
            bar = ax[lab].pcolormesh(lon, lat, seasonal_means[j], transform = data_proj, cmap = cmocean.cm.thermal, vmin = 0, vmax = 1)
            ax[lab].set_title(f"Confidence for {season}")
            ax[lab].text(0.0, 1.0, f"{lab})", transform=ax[lab].transAxes + trans,
                fontsize='medium', va='bottom')

        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.suptitle(f'Seasonal confidence for fast-ice ({cont}) contour')
        fig.colorbar(bar, cax=cbar_ax)
        fig.savefig(f'/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/{model_name}/confidence_test_contour_{i}.png')
    
    total = np.concatenate((djf, mam, jja, son), axis = 0)
    total_means = total.mean(axis = 0)

    labels += ['e', 'f']
    fig, ax = plt.subplot_mosaic('''
                                 ab
                                 cd
                                 ef
                                 ''',
                                 figsize = (10, 15),
                                 subplot_kw={'projection': map_proj})
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)


    for i, lab, cont in zip(range(1, 7), labels, contours[1:]):
        bar = ax[lab].pcolormesh(lon, lat, total_means[i], transform = data_proj, cmap = cmocean.cm.thermal, vmin = 0, vmax = 1)
        ax[lab].set_title(f"Confidence for contour {cont}")
        ax[lab].text(0.0, 1.0, f"{lab})", transform=ax[lab].transAxes + trans,
            fontsize='medium', va='bottom')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(bar, cax=cbar_ax)
    fig.savefig(f'/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/{model_name}/confidence_mean_annual.png')

if __name__ == "__main__":
    main()
