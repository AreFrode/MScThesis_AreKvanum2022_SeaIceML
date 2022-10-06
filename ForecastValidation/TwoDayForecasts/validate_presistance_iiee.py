import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from two_day_forecast.createHDF import onehot_encode_sic
from verification_metrics import *
from tqdm import tqdm
from matplotlib import pyplot as plt


def main():
    PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/figures/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    data_2021 = sorted(glob.glob(f"{PATH_PERSISTANCE}2021/**/*.hdf5", recursive = True))

    output_df = pd.DataFrame(columns = ['date', 'IIEE', 'a_plus', 'a_minus'])

    for sample in tqdm(data_2021):
        date = sample[-13:-5]
        save_location = f"{PATH_FIGURES}{date[:4]}/{date[4:6]}/"

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        with h5py.File(sample, 'r') as infile:
            sic = onehot_encode_sic(infile['sic'][450:, :1840])
            sic_target = infile['sic_target'][450:, :1840]
            lsmask = infile['lsmask'][450:, :1840]

        a_plus, a_minus, ocean, ice, = IIEE(sic, sic_target, lsmask)
        iiee = a_plus.sum() + a_minus.sum()

        fig = plt.figure(figsize = (20,20))
        ax = plt.axes()

        a_plus_masked = np.ma.masked_array(a_plus, a_plus == 0)
        a_minus_masked = np.ma.masked_array(a_minus, a_minus == 0)
        ocean_masked = np.ma.masked_array(ocean, ocean == 0)
        ice_masked = np.ma.masked_array(ice, ice == 0)

        ax.set_title(f"IIEE persistance - one day target initiated {date}", fontsize=30)
        ax.pcolormesh(a_plus_masked, zorder=3, cmap='summer')
        ax.pcolormesh(a_minus_masked, zorder=3, cmap='Reds_r')
        ax.pcolormesh(ocean_masked, zorder=3, cmap='Blues_r')
        ax.pcolormesh(ice_masked, zorder=3, cmap='cool')


        plt.savefig(f"{save_location}{date}_iiee.png")
        ax.cla()
        plt.close(fig)

        tmp_df = pd.DataFrame([[pd.to_datetime(date, format="%Y%m%d"), iiee, a_plus.sum(), a_minus.sum()]], columns = ['date', 'IIEE', 'a_plus', 'a_minus'])
        output_df = pd.concat([output_df, tmp_df])

        del a_plus
        del a_plus_masked
        del a_minus
        del a_minus_masked
        del ocean
        del ocean_masked
        del ice
        del ice_masked

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}test_iiee.csv")
    


if __name__ == "__main__":
    main()