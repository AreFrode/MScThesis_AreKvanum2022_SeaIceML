import os
import glob
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from createHDF import onehot_encode_sic
from netCDF4 import Dataset

def read_data(path_counts):
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/"
    icecharts = sorted(glob.glob(f"{PATH_DATA}**/*.nc"))

    data_list = []

    for ic in icecharts:
        yyyymmdd = ic[-17:-9]
        print(yyyymmdd, end='\r')
    
        with Dataset(ic, 'r') as nc:
            sic = onehot_encode_sic(nc.variables['sic'][578:,:1792])

        idxs, counts = np.unique(sic, return_counts = True)
    
        data_list.append([yyyymmdd, *counts])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['0', '1', '2', '3', '4', '5', '6'])
    df.to_csv(path_counts)

def read_amsr2(path_counts):
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2/2022/"
    
    with Dataset("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/ml/2022/01/weights_05011118_20220105_b20220103.nc", 'r') as nc:
        arome_lsmask = nc.variables['lsmask'][:]

    
    amsr_data = sorted(glob.glob(f"{PATH_DATA}**/*.nc"))

    with Dataset(amsr_data[0], 'r') as nc:
        amsr2_lsmask = nc.variables['lsmask'][:]

    lsmask = arome_lsmask + amsr2_lsmask
    lsmask[lsmask > 1] = 1

    data_list = []

    for amsr in amsr_data:
        yyyymmdd = amsr[-11:-3]
        print(yyyymmdd, end='\r')

        with Dataset(amsr, 'r') as nc:
            sic = np.where(lsmask == 1, -1, nc.variables['sic'][:])

        idxs, counts = np.unique(sic, return_counts = True)

        data_list.append([yyyymmdd, *counts[1:]])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['0', '1', '2', '3', '4', '5', '6'])

    df.to_csv(path_counts)



def main():
    path_counts = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_amsr2_counts.csv'
    if not os.path.exists(path_counts):
        read_amsr2(path_counts)

    df = pd.read_csv(path_counts, index_col = 0)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')

    months = df.resample('M').mean()

    data = []

    for i in range(len(months)):
        data.append([])
        current_month = months.iloc[i]
        month_total = current_month.sum()
        for j in range(len(current_month)):
            data[i].append(current_month.iloc[j] / month_total)

    df_frac = pd.DataFrame(data, columns = ['0', '1', '2', '3', '4', '5', '6'], index = months.index)

    sns.set_context('paper')
    sns.set_theme()
    
    df_frac = df_frac.set_index(np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Des']))

    ax = df_frac.plot(kind = 'bar', stacked = True, figsize = (14,8), rot = 0, fontsize = 20)
    ax.set_xlabel('Month', fontsize = 18)
    ax.set_ylabel('Fraction', fontsize = 18)
    ax.set_title('2022 Monthly Sea Ice area fraction AMSR2', fontsize = 18)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], ['Ice Free Open Water', 'Open Water', 'Very Open Drift Ice', 'Open Drift Ice', 'Close Drift Ice', 'Very Close Drift Ice', 'Fast Ice'][::-1], fontsize = 18)
    plt.savefig('2022-sic-distribution_amsr2.png')

    

if __name__ == "__main__":
    main()