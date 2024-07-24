# Date: 13.05.24
# Author: Are Frode Kvanum
# HEAVILY inspired by: https://github.com/cerea-daml/nextsim-surrogate/blob/v1/compute_PSD.py (Durand2024)

import glob
import h5py

import numpy as np
import scipy.stats as stats

from matplotlib import pyplot as plt

def computePSD(field, Npix = 1792):
    PSDtot = np.zeros((Npix // 2))

    fourier_image = np.fft.fft2(field)
    fourier_amps = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(Npix) * Npix

    kreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kreq2D[0] ** 2+ kreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amps = fourier_amps.flatten()
    kbins = np.arange(0.5, Npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm,
                                          fourier_amps,
                                          statistic = "mean",
                                          bins = kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    return Abins



def main():
    # Define paths and constants
    path_input_icecharts = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/noTarget/lead_time_1/"

    path_models = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/"

    models = ['weights_08031256', 'weights_21021550', 'weights_09031047']

    x = np.arange(1, 1793 // 2 + 1)
    Nres = 1792
    Res = 25
    resolution = 1 / x * Nres * Res

    # print(resolution.shape)
    # exit()


    with h5py.File(sorted(glob.glob(f"{path_input_icecharts}2022/01/*"))[3], 'r') as ic0:
        test_icechart = ic0['sic'][578:,:1792]

    test_psd = computePSD(test_icechart)

    with h5py.File(sorted(glob.glob(f"{path_models}{models[0]}/2022/01/*"))[2], 'r') as ic1:
        test_pred_1 = ic1['y_pred'][0]

    test_pred_1_psd = computePSD(test_pred_1)

    with h5py.File(sorted(glob.glob(f"{path_models}{models[2]}/2022/01/*"))[0], 'r') as ic1:
        test_pred_2 = ic1['y_pred'][0]

    test_pred_2_psd = computePSD(test_pred_2)

    fig, ax = plt.subplots()
    ax.loglog(resolution, test_psd, '.', label = 'IceChart')
    ax.loglog(resolution, test_pred_1_psd, '.', label = 'DeepLearning t+24h')
    ax.loglog(resolution, test_pred_2_psd, '.', label = 'DeepLearning t+72h')
    ax.invert_xaxis()

    ax.legend()
    fig.savefig('testfig.png')


if __name__ == "__main__":
    main()