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
# from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from netCDF4 import Dataset
from createHDF import onehot_encode_sic
from scipy.interpolate import NearestNDInterpolator
from verification_metrics import find_ice_edge, ice_edge_length, IIEE_alt as IIEE


from datetime import datetime, timedelta
from helper_functions import read_config_from_csv

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def main():
    plt.rcParams['axes.grid'] = False
    
    # plt.rcParams['pdf.compression'] = True
    # plt.rcParams['path.simplify'] = True
    # plt.rcParams['axes.grid'] = False
    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]

    path_pred = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"
    path_raw = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    config = read_config_from_csv(f"{path_pred[:-5]}configs/{weights}.csv")

    path = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/"
    # path_figure = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/lustre_poster_extra/"
    path_figure = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/lustre_poster_extra/"


    lower_boundary = 578
    rightmost_boundary = 1792

    year = '2022'
    month = '03'
    bday = '23'
    vday = '25'

    bdate = datetime.strptime(f"{year}{month}{bday}", '%Y%m%d')
    vdate = datetime.strptime(f"{year}{month}{vday}", '%Y%m%d')

    data_ml = glob.glob(f"{path_pred}{weights}/{year}/{month}/SIC_UNET_v{year}{month}{vday}_b{year}{month}{bday}T15Z.hdf5")[0]

    data_ict0 = glob.glob(f"{path_raw}{year}/{month}/ICECHART_1kmAromeGrid_{year}{month}{bday}T1500Z.nc")[0]
    data_ict2 = glob.glob(f"{path_raw}{year}/{month}/ICECHART_1kmAromeGrid_{year}{month}{vday}T1500Z.nc")[0]


    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    PRJ = pyproj.Proj(map_proj.proj4_init)
    data_proj = ccrs.PlateCarree()

    # xticks = [-20,-10, 0, 10,20,30,40,50,60,70,80,90,100,110,120]
    xticks = [-10, 0, 10, 20, 30, 40, 50, 60, 70]
    # yticks = [60,65,70, 75, 80, 85,90]
    yticks = [70, 75, 80, 85]

    cividis = mpl.colormaps['cividis']
    ice_colors = cmocean.cm.ice

    preferred_cmap = ice_colors(np.linspace(0, 1, config['num_outputs']))

    # newcolors[0, :-1] = np.array([34., 193., 224.]) / 255.
    # newcolors[0, -1] = 0.3
    # newcolors[1:] = preferred_cmap[2:]
    ice_cmap = WMOcolors.cm.sea_ice_chart()
    newcolors = ice_cmap(np.linspace(0, 1, 7))
    newcolors = np.delete(newcolors, obj=1, axis=0)
    ice_cmap = mcolors.ListedColormap(newcolors)

    ice_levels = np.linspace(0, config['num_outputs'] - 1, config['num_outputs'], dtype = 'int')
    ice_norm = mcolors.BoundaryNorm(ice_levels, ice_cmap.N)

    # Colorblind friendly land
    # land_color = mpl.colormaps['autumn'](np.linspace(0,1,2))
    # land_color[:, :-1] = np.array([211., 95., 183.]) / 255.
    land_cmap = WMOcolors.cm.land()

    if config['reduced_classes']:
        ice_ticks = ['0', '10 - 40', '40 - 70', '70 - 90', '90 - 100']

    else:
        ice_ticks = ['0', '0 - 10', '10 - 40', '40 - 70', '70 - 90', '90 - 100', '100']

    ice_ticks = ['0', '10 - 30', '40 - 60', '70 - 80', '90 - 100', 'fast ice']

    sns.set_theme(context = 'talk')
    figsize = (20,8)

    # fig, ax = plt.subplots(ncols = 3, nrows=2, figsize=(25,10), constrained_layout = True, subplot_kw={'projection': map_proj})
    fig = plt.figure(figsize = figsize, constrained_layout = True)
    ax = []

    ax.append(fig.add_subplot(131, projection = map_proj))
    ax.append(fig.add_subplot(132, projection = map_proj))
    ax.append(fig.add_subplot(133, projection = map_proj))
    # ax.append(fig.add_subplot(235))


    # ax = plt.axes(projection = map_proj)

    with h5py.File(sorted(glob.glob(f"{path}01/*.hdf5"))[0], 'r') as f:
        lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
        lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:622, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    x0,y0 = PRJ(lon[0,0], lat[0,0])
    x1,y1 = PRJ(lon[-1,-1], lat[-1,-1])

    # Plot figure a
    with Dataset(data_ict0, 'r') as ic_0:
        sic0 = onehot_encode_sic(ic_0['sic'][lower_boundary:, :rightmost_boundary])
    
    sic0_interpolator = NearestNDInterpolator(mask_T, sic0[mask])
    sic0 = sic0_interpolator(*np.indices(sic0.shape))

    ice_edge0 = find_ice_edge(sic0, lsmask)

    sic0 = np.where(sic0 == 1, 0, sic0)
    sic0 = np.where(sic0 > 0, sic0 - 1, sic0)

    ax[0].pcolormesh(lon, lat, sic0, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1, rasterized = True)
    ax[0].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap=land_cmap, rasterized = True)
    ax[0].scatter(lon, lat, 0.05*ice_edge0, transform=data_proj, zorder=3, color='black', rasterized = True)


    ax[0].set_title(f"Sea ice chart {bdate.strftime('%d')}rd {bdate.strftime('%B')} {bdate.strftime('%Y')}")

    ax[0].set_xlim(x0,x1)
    ax[0].set_ylim(y0,y1)
    
    fig.canvas.draw()
    # ax[0].gridlines(draw_labels = True, xlocs = xticks, ylocs = yticks, rotate_labels = False, color = 'dimgrey')
    ax[0].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')

    # ax[0].tick_params(axis='both', direction = 'out', right = False, top = False)

    # ax[0].set_xticks(xticks, crs = data_proj)
    # ax[0].set_yticks(yticks, crs = data_proj)
    ax[0].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[0].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[0], xticks)
    LambertLabels.lambert_yticks(ax[0], yticks)

    # divider0 = make_axes_locatable(ax[0])
    # cax0 = divider0.append_axes("bottom", size="3%", pad=.05)
    # cax0.axis('off')

    ax[0].set_frame_on(False)

    # Plot figure b
    with Dataset(data_ict2, 'r') as ic_2:
        sic2 = onehot_encode_sic(ic_2['sic'][lower_boundary:, :rightmost_boundary])
    
    sic2_interpolator = NearestNDInterpolator(mask_T, sic2[mask])
    sic2 = sic2_interpolator(*np.indices(sic2.shape))

    ice_edge2 = find_ice_edge(sic2, lsmask)

    sic2 = np.where(sic2 == 1, 0, sic2)
    sic2 = np.where(sic2 > 0, sic2 - 1, sic2)

    # iiee = IIEE(sic0, sic2, lsmask, threshold = 1)

    ax[1].pcolormesh(lon, lat, sic2, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1, rasterized = True)
    ax[1].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap=land_cmap, rasterized = True)
    ax[1].scatter(lon, lat, 0.05*ice_edge0, transform=data_proj, zorder=3, color='black', rasterized = True)

    # ax[1].pcolormesh(lon, lat, np.ma.masked_less(iiee[0],1),alpha=0.7, transform=data_proj, cmap = 'summer', zorder=3, rasterized = True)
    # ax[1].pcolormesh(lon, lat, np.ma.masked_less(iiee[1], 1),alpha=0.7, transform=data_proj, zorder=4, cmap='winter', rasterized = True)

    ax[1].set_xlim(x0,x1)
    ax[1].set_ylim(y0,y1)

    ax[1].set_title(f"Sea ice chart {vdate.strftime('%d')}th {vdate.strftime('%B')} {vdate.strftime('%Y')}")
    
    fig.canvas.draw()
    ax[1].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[1].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[1].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[1], xticks)
    LambertLabels.lambert_yticks(ax[1], yticks)

    ax[1].set_frame_on(False)

    # divider1 = make_axes_locatable(ax[1])
    # cax1 = divider1.append_axes("bottom", size="3%", pad=.05)

    '''
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    '''
    
    # Plot figure c
    with h5py.File(data_ml, 'r') as ml:
        sicml = ml['y_pred'][0,:,:]

    sicml = np.where(sicml == 1, 0, sicml)
    sicml = np.where(sicml > 0, sicml - 1, sicml)

    # iiee = IIEE(sicml, sic2, lsmask, threshold = 1)
    # iiee_p = IIEE(sic0, sic2, lsmask, threshold = 1)
    
    # print(iiee[0].sum() + iiee[1].sum())
    # print(ice_edge2)
    # print(ice_edge_length(ice_edge2, s = 1))

    # print(f"niiee ml {(iiee[0].sum() + iiee[1].sum()) / ice_edge_length(ice_edge2, s = 1)}")
    # print(f"niiee persistence {(iiee_p[0].sum() + iiee_p[1].sum()) / ice_edge_length(ice_edge2, s = 1)}")



    ax[2].pcolormesh(lon, lat, sicml, transform=data_proj, norm = ice_norm, cmap = ice_cmap, zorder=1, rasterized = True)
    ax[2].pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, zorder=2, cmap=land_cmap, rasterized = True)

    ax[2].scatter(lon, lat, 0.05*ice_edge0, transform=data_proj, zorder=4, color='black', rasterized = True)
    ax[2].scatter(lon[::], lat[::], 0.05*ice_edge2[::], transform=data_proj, zorder=3, color='dodgerblue', rasterized=True)

    # ax[2].pcolormesh(lon, lat, np.ma.masked_less(iiee[0],1),alpha=0.7, transform=data_proj, cmap = 'summer', zorder=3, rasterized = True)
    # ax[2].pcolormesh(lon, lat, np.ma.masked_less(iiee[1], 1),alpha=0.7, transform=data_proj, zorder=4, cmap='winter', rasterized = True)

    ax[2].set_xlim(x0,x1)
    ax[2].set_ylim(y0,y1)

    ax[2].set_title(f"Deep learning {vdate.strftime('%d')}th {vdate.strftime('%B')} {vdate.strftime('%Y')}, initialized {bdate.strftime('%d')}rd {bdate.strftime('%B')} {bdate.strftime('%Y')}")
    
    fig.canvas.draw()
    ax[2].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[2].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[2].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[2], xticks)
    LambertLabels.lambert_yticks(ax[2], yticks)

    ax[2].set_frame_on(False)

    # divider2 = make_axes_locatable(ax[2])
    # cax2 = divider2.append_axes("bottom", size="3%", pad=.05)
    # cax2.axis('off')

    '''
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    '''

    mapper = mpl.cm.ScalarMappable(cmap = ice_cmap, norm = ice_norm)
        # mapper.set_array([-1, 8])

    ax[1].grid(False)
    cbar = fig.colorbar(mapper,
                        ax = ax[1],
                        spacing = 'uniform',
                        orientation = 'horizontal',
                        pad = .0,
                        drawedges = True
                        # fraction = 0.07
    )

    cbar.set_label(label = 'WMO sea ice concentration intervals [%]')#, size = 16)
    cbar.set_ticks(ice_levels[:-1] + .5, labels = ice_ticks)
    cbar.outline.set_linewidth(1.2)
    cbar.dividers.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')
    cbar.dividers.set_edgecolor('black')
    cbar.minorticks_off()
    
        
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(3), ['a', 'b', 'c']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, va='bottom')

    # cbar.ax.tick_params(labelsize = 16)

    # plt.tight_layout()
    print('saving fig')
    fig.savefig(f"{path_figure}new_predictions.png")

    exit()
    with h5py.File(f'{path}{month}/PreparedSample_v{vdate.strftime("%Y%m%d")}_b{bdate.strftime("%Y%m%d")}.hdf5', 'r') as arome:
        t2m = arome['t2m'][lower_boundary:, :rightmost_boundary]
        xwind = arome['xwind'][lower_boundary:, :rightmost_boundary]
        ywind = arome['ywind'][lower_boundary:, :rightmost_boundary]

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    ax = []

    ax.append(fig.add_subplot(131, projection = map_proj))
    ax.append(fig.add_subplot(132, projection = map_proj))
    ax.append(fig.add_subplot(133, projection = map_proj))

    t2m_bar = ax[0].pcolormesh(lon, lat, t2m, cmap = cmocean.cm.thermal, transform=data_proj, rasterized = True)
    ax[0].set_title(f"2-meter temperature")

    ax[0].set_xlim(x0,x1)
    ax[0].set_ylim(y0,y1)
    
    fig.canvas.draw()
    ax[0].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[0].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[0].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[0], xticks)
    LambertLabels.lambert_yticks(ax[0], yticks)

    cbar_t2m = fig.colorbar(t2m_bar, ax = ax[0], orientation = 'horizontal')
    cbar_t2m.set_label('Temperature [K]')


    # Plot figure b
    # iiee = IIEE(sic0, sic2, lsmask, threshold = 1)

    xwind_bar = ax[1].pcolormesh(lon, lat, xwind, cmap = 'winter', transform=data_proj, rasterized = True)


    ax[1].set_xlim(x0,x1)
    ax[1].set_ylim(y0,y1)

    ax[1].set_title(f"x-wind")
    
    fig.canvas.draw()
    ax[1].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[1].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[1].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[1], xticks)
    LambertLabels.lambert_yticks(ax[1], yticks)

    cbar_xwind = fig.colorbar(xwind_bar, ax = ax[1], orientation = 'horizontal')
    cbar_xwind.set_label('x-wind [m/s]')

    # divider1 = make_axes_locatable(ax[1])
    # cax1 = divider1.append_axes("bottom", size="3%", pad=.05)

    # Plot figure c
    ywind_bar = ax[2].pcolormesh(lon, lat, ywind, cmap = 'winter', transform=data_proj, rasterized = True)

    ax[2].set_xlim(x0,x1)
    ax[2].set_ylim(y0,y1)

    ax[2].set_title(f"y-wind")
    
    fig.canvas.draw()
    ax[2].gridlines(xlocs = xticks, ylocs = yticks, color = 'dimgrey')
    ax[2].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax[2].yaxis.set_major_formatter(LATITUDE_FORMATTER)
    LambertLabels.lambert_xticks(ax[2], xticks)
    LambertLabels.lambert_yticks(ax[2], yticks)

    cbar_ywind = fig.colorbar(ywind_bar, ax = ax[2], orientation = 'horizontal')
    cbar_ywind.set_label('y-wind [m/s]')

    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)    
    for i,j in zip(range(3), ['a', 'b', 'c']):
        ax[i].set_anchor('N')
        ax[i].text(0.0, 1.0, f"{j})", transform = ax[i].transAxes + trans, fontsize='medium', va='bottom')

    fig.savefig(f"{path_figure}atmosphere.png")

    cmapmin = mcolors.ListedColormap(['lightcoral'])
    cmapplus = mcolors.ListedColormap(['deepskyblue'])

    # ice = mcolors.ListedColormap([cmocean.cm.ice(1)])
    # water = mcolors.ListedColormap([cmocean.cm.ice(0)])
    

    fig = plt.figure(constrained_layout = True)
    ax = fig.add_subplot(111, projection=map_proj)

    
    ax.pcolormesh(lon, lat, np.where(iiee[0] == 1, 1, np.nan), cmap = cmapplus, transform = data_proj)
    ax.pcolormesh(lon, lat, np.where(iiee[1] == 1, 1, np.nan), cmap = cmapmin, transform = data_proj)
    ax.pcolormesh(lon, lat, np.where(iiee[2] == 1, 1, np.nan), cmap = cmocean.cm.ice, transform = data_proj)
    # ax.pcolormesh(lon, lat, np.where(iiee[3] == 1, 1, np.nan), cmap = cmocean.cm.ice_r, transform = data_proj)
    ax.pcolormesh(lon, lat, np.ma.masked_less(lsmask, 1), transform=data_proj, cmap=land_cmap)
    
    cmap = mcolors.ListedColormap([cmocean.cm.ice(0), cmocean.cm.ice_r(0), 'lightcoral', 'deepskyblue'])

    mapper2 = mpl.cm.ScalarMappable(cmap = cmap)
    tick_levels = [0.125 + i*0.25 for i in range(4)]
    tick_labels = ['Ocean', 'Sea ice','Under', 'Over']

    cbar2 = fig.colorbar(mapper2, ax = ax, spacing = 'uniform', orientation = 'vertical', drawedges = True)
    cbar2.set_ticks(tick_levels, labels = tick_labels)
    cbar2.outline.set_linewidth(2)
    cbar2.dividers.set_linewidth(2)
    cbar2.outline.set_edgecolor('black')
    cbar2.dividers.set_edgecolor('black')

    fig.savefig(f'{path_figure}iiee.png')

if __name__ == "__main__":
    main()
