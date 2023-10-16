import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")

import h5py
import glob
import os
import LambertLabels
import pyproj

import numpy as np
import matplotlib as mpl
import seaborn as sns
import cmocean
import WMOcolors

from matplotlib import pyplot as plt, transforms as mtransforms, colors as mcolors
from cartopy import crs as ccrs
from shapely.errors import ShapelyDeprecationWarning
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from netCDF4 import Dataset
from verification_metrics import find_ice_edge



from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    plt.rcParams['axes.grid'] = False
    
    # plt.rcParams['pdf.compression'] = True
    # plt.rcParams['path.simplify'] = True
    # plt.rcParams['axes.grid'] = False

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/2022/"
    # path_figure = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/lustre_poster_extra/"
    path_figure = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/seg-grad-cam/thesis_figs/"
    PATH_DATA = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/"

    data_2022 = np.array(sorted(glob.glob(f"{PATH_DATA}2022/**/*.hdf5")))

    lower_boundary = 578
    rightmost_boundary = 1792

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))

    data_proj = ccrs.PlateCarree()

    xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    yticks = [60,65,70, 75, 80, 85,90]

    with h5py.File(sorted(glob.glob(f"{path}01/*.hdf5"))[0], 'r') as f:
        lat = f['lat'][lower_boundary:, :rightmost_boundary]
        lon = f['lon'][lower_boundary:, :rightmost_boundary]
        lsmask = f['lsmask'][lower_boundary:, :rightmost_boundary]


    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()
    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    land_cmap = WMOcolors.cm.land()

    sns.set_theme(context = 'talk')
    figsize = (6,6)

    with Dataset('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/seg-grad-cam/figure_1.nc', 'r') as fig1:
        cam1 = fig1['cam'][:]

    with h5py.File(data_2022[0], 'r') as h:
        sample_0 = h['sic'][lower_boundary:, :rightmost_boundary]
        
    # Figure I
    contours = ['>=10%', '>=40%', '>=70%', '>=90%']
    edges_1 = [find_ice_edge(sample_0, lsmask, i+2) for i in range(4)]

    fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection' : map_proj}, figsize = figsize, constrained_layout = True)
    ax = ax.reshape(-1)

    fig.suptitle('seg-grad-cam default Deep learning model \n for varying contours 5th Jan', fontsize = 18)
    for i in range(4):
        cbar = ax[i].pcolormesh(lon, lat, np.where(cam1[i] > 0, cam1[i], np.nan), cmap = 'jet', transform=data_proj, rasterized = True)
        ax[i].scatter(lon, lat, 0.05*edges_1[i], transform = data_proj, color = 'black')
        ax[i].set_title(f"{contours[i]} contour")

        ax[i].set_xlim(x0,x1)
        ax[i].set_ylim(y0,y1)
    

    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(4), ['a', 'b', 'c', 'd']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')


    colorbar = fig.colorbar(cbar, ax=ax.ravel().tolist(), shrink = 0.88)
            
    colorbar.mappable.set_clim(0, 1)
    colorbar.set_ticks([])

    fig.savefig(f"{path_figure}baseline_contours.png")

    with Dataset('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/seg-grad-cam/figure_2.nc', 'r') as fig2:
        cam2 = fig2['cam'][:]

    # Figure II
    fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection' : map_proj}, figsize = figsize, constrained_layout = True)
    ax = ax.reshape(-1)

    fig.suptitle('seg-grad-cam reduced classes model \n for varying contours 5th Jan', fontsize = 18)
    for i in range(4):
        cbar = ax[i].pcolormesh(lon, lat, np.where(cam2[i] > 0, cam2[i], np.nan), cmap = 'jet', transform=data_proj, rasterized = True)
        ax[i].scatter(lon, lat, 0.05*edges_1[i], transform = data_proj, color = 'black')
        ax[i].set_title(f"{contours[i]} contour")

        ax[i].set_xlim(x0,x1)
        ax[i].set_ylim(y0,y1)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(4), ['a', 'b', 'c', 'd']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')

    colorbar = fig.colorbar(cbar, ax=ax.ravel().tolist(), shrink = 0.88)
    
    colorbar.mappable.set_clim(0, 1)
    colorbar.set_ticks([])

    fig.savefig(f"{path_figure}reduced_classes_contours.png")

    with Dataset('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/seg-grad-cam/figure_3.nc', 'r') as fig3:
        cam3 = fig3['cam'][:]

    dates = ['3rd March', '3rd June', '7th September', '7th December']

    # Figure III
    samples = [24, 59, 98, 136]

    edges_2 = [find_ice_edge(h5py.File(data_2022[samples[i]], 'r')['sic'][lower_boundary:, :rightmost_boundary], lsmask, 2) for i in range(4)]

    fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection' : map_proj}, figsize = figsize, constrained_layout = True)
    ax = ax.reshape(-1)

    fig.suptitle('seg-grad-cam default Deep learning model \n for varying dates', fontsize = 18)
    for i in range(4):
        cbar = ax[i].pcolormesh(lon, lat, np.where(cam3[i] > 0, cam3[i], np.nan), cmap = 'jet', transform=data_proj, rasterized = True)
        ax[i].scatter(lon, lat, 0.05*edges_2[i], transform = data_proj, color = 'black')
        ax[i].set_title(f"{dates[i]}")

        ax[i].set_xlim(x0,x1)
        ax[i].set_ylim(y0,y1)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(4), ['a', 'b', 'c', 'd']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')

    colorbar = fig.colorbar(cbar, ax=ax.ravel().tolist(), shrink = 0.88)
    
    colorbar.mappable.set_clim(0, 1)
    colorbar.set_ticks([])

    fig.savefig(f"{path_figure}baseline_dates.png")

    with Dataset('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/seg-grad-cam/figure_4.nc', 'r') as fig4:
        cam4 = fig4['cam'][:]

    # Figure IV
    fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection' : map_proj}, figsize = figsize, constrained_layout = True)
    ax = ax.reshape(-1)

    fig.suptitle('seg-grad-cam Deep learning w.o. t2m \n for varying dates', fontsize = 18)
    for i in range(4):
        cbar = ax[i].pcolormesh(lon, lat, np.where(cam4[i] > 0, cam4[i], np.nan), cmap = 'jet', transform=data_proj, rasterized = True)
        ax[i].scatter(lon, lat, 0.05*edges_2[i], transform = data_proj, color = 'black')
        ax[i].set_title(f"{dates[i]}")

        ax[i].set_xlim(x0,x1)
        ax[i].set_ylim(y0,y1)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(4), ['a', 'b', 'c', 'd']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')

    colorbar = fig.colorbar(cbar, ax=ax.ravel().tolist(), shrink = 0.88)
    
    colorbar.mappable.set_clim(0, 1)
    colorbar.set_ticks([])

    fig.savefig(f"{path_figure}not2m_dates.png")

    # Figure V



    fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection' : map_proj}, figsize = figsize, constrained_layout = True)
    ax = ax.reshape(-1)

    fig.suptitle('t2m predictor fields', fontsize = 18)
    for i in range(4):
        with h5py.File(data_2022[samples[i]], 'r') as preds:
            t2m = preds['t2m'][lower_boundary:, :rightmost_boundary]

        cbar = ax[i].pcolormesh(lon, lat, t2m, cmap = cmocean.cm.thermal, transform=data_proj, rasterized = True)
        ax[i].scatter(lon, lat, 0.05*edges_2[i], transform = data_proj, color = 'black')
        fig.colorbar(cbar, ax=ax[i], orientation = 'horizontal')

        ax[i].set_title(f"{dates[i]}")

        ax[i].set_xlim(x0,x1)
        ax[i].set_ylim(y0,y1)
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(4), ['a', 'b', 'c', 'd']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')

    
    fig.savefig(f"{path_figure}t2m.png")



if __name__ == "__main__":
    main()