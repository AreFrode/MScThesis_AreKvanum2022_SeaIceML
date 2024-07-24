import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures')

import glob
import h5py
import WMOcolors
import cmocean

import numpy as np
from verification_metrics import find_ice_edge, ice_edge_length
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from scipy import fft


def fft_filter(img, kernel):
    sz = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])

    kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')

    kernel = fft.ifftshift(kernel)

    img_fft = fft.fft2(img)
    kernel_fft = fft.fft2(kernel)

    smoothed_img = img_fft * kernel_fft
    return fft.ifft2(smoothed_img).real

def create_mean_filter(side_length):
    kernel = np.ones((side_length, side_length)) / (side_length**2)

    return kernel


def main():
    # Define paths and constants
    path_input_icecharts = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/noTarget/lead_time_1/"

    path_models = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"

    models = ['weights_08031256', 'weights_21021550', 'weights_09031047']

    map_proj = ccrs.LambertConformal(central_latitude = 77.5,
                                     central_longitude = -25,
                                     standard_parallels = (77.5, 77.5))
    data_proj = ccrs.PlateCarree()

    ice_cmap = cmocean.cm.ice
    ice_cmap = WMOcolors.cm.sea_ice_chart()
    land_cmap = WMOcolors.cm.land()

    with h5py.File(sorted(glob.glob(f"{path_input_icecharts}2022/01/*"))[3], 'r') as ic0:
        test_icechart = ic0['sic'][578:,:1792]
        lsmask = ic0['lsmask'][578:,:1792]
        lon = ic0['lon'][578:,:1792]
        lat = ic0['lat'][578:,:1792]

    lsmask[:15, :] = 1
    lsmask[-15:, :] = 1
    lsmask[:, :15] = 1
    lsmask[:, -15:] = 1

    fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
    ax.pcolormesh(lon, lat, lsmask, transform = data_proj, cmap = land_cmap)

    fig.savefig('lsmask.png')




    # ice_chart_edge = find_ice_edge(test_icechart, lsmask, verbose = True)
    # ice_chart_edge_length = ice_edge_length(ice_chart_edge)

    # print(ice_chart_edge_length)

    smooth_constant = 7

    # test_icechart = np.pad(test_icechart, ((0,smooth_constant - 1), (0,smooth_constant - 1)), 'edge')

    mean_filter = create_mean_filter(smooth_constant)

    ice_edge_length_charts = []
    ice_chart_edge = find_ice_edge(test_icechart, lsmask, verbose = True)
    ice_edge_length_charts.append(ice_edge_length(ice_chart_edge))

    fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
    ax.pcolormesh(lon, lat, test_icechart, transform = data_proj, cmap = ice_cmap)
    # ax[1].pcolormesh(lon, lat, ice_chart_edge, transform = data_proj, cmap = ice_cmap)

    fig.savefig('Unsmoothed.png')

    for i in range(10):
        # Ice charts are categorical, this is preserved after smoothing by rounding
        test_icechart = fft_filter(test_icechart, mean_filter)
        rounded_icechart = np.rint(test_icechart)
        ice_chart_edge = find_ice_edge(rounded_icechart, lsmask, verbose = True)
        ice_edge_length_charts.append(ice_edge_length(ice_chart_edge))

        fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
        ax.pcolormesh(lon, lat, rounded_icechart, transform = data_proj, cmap = ice_cmap)
        # ax[1].pcolormesh(lon, lat, ice_chart_edge, transform = data_proj, cmap = land_cmap)

        fig.savefig(f'iter{i}.png')

 
    test_icechart = np.where(lsmask == 1, 0, test_icechart)
    fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
    ax.pcolormesh(lon, lat, test_icechart, transform = data_proj, cmap = ice_cmap)
    # ax.pcolormesh(lon, lat, smoothed_test_icechart[(smooth_constant-1)//2+1:-(smooth_constant-1)//2,(smooth_constant-1)//2+1:-(smooth_constant-1)//2], transform = data_proj, cmap = ice_cmap)

    fig.savefig(f'{smooth_constant}x{smooth_constant}-smoothed.png')




    with h5py.File(sorted(glob.glob(f"{path_models}{models[0]}/2022/01/*"))[2], 'r') as ic1:
        test_pred_1 = ic1['y_pred'][0]
    
    fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
    ax.pcolormesh(lon, lat, test_pred_1, transform = data_proj, cmap = ice_cmap)

    fig.savefig('Pred.png')

    pred_edge = find_ice_edge(test_pred_1, lsmask, verbose = True)
    

    fig, ax = plt.subplots()
    ax.plot(ice_edge_length_charts, 'o-', label = 'Ice chart ice edge length')
    ax.hlines(ice_edge_length(pred_edge), xmin = 0, xmax = len(ice_edge_length_charts), linestyles = 'dashed', colors = 'r', label = 'Deep learning ice edge length')

    fig.legend()
    ax.set_xlabel('Smoothing iterations')
    ax.set_ylabel('Ice edge length [km]')

    fig.savefig('smoothing-iceedge.pdf')
    # ml_edge = find_ice_edge(test_pred_1, lsmask, verbose = True)
    # ml_edge_length = ice_edge_length(ml_edge)

    # print(ml_edge_length)
    



if __name__ == "__main__":
    main()