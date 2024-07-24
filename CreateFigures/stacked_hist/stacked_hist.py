import os
import glob
import sys
import h5py
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from createHDF import onehot_encode_sic
from netCDF4 import Dataset

def read_data(path_counts, path = None):
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

def read_data_seasons(path_counts, path = None):
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/"
    icecharts = sorted(glob.glob(f"{PATH_DATA}**/*.nc"))

    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"

    with Dataset(f"{path_arome}2019/01/AROME_1kmgrid_20190101T18Z.nc") as constants:
        lsmask = constants['lsmask'][:,:-1]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp

    mask = lsmask + baltic_mask
    mask[mask > 1] = 1

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()

    data_list = []

    for ic in icecharts:
        yyyymmdd = ic[-17:-9]
        print(yyyymmdd, end='\r')
    
        with Dataset(ic, 'r') as nc:
            sic = onehot_encode_sic(nc.variables['sic'][:,:-1])
            sic = np.where(mask == 1, -1, sic)

        sic = sic[578:, :1792]
        idxs, counts = np.unique(sic, return_counts = True)
    
        data_list.append([f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}", *counts[1:]])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['IFOW', 'OW', 'VODI', 'ODI', 'CDI', 'VCDI', 'FI'])

    for j, idx in zip(range(len(months) - 1), meteorological_seasons):
                df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]


    df.to_csv(path_counts)

def read_ml(path_counts, path):
    # PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/"
    icecharts = sorted(glob.glob(f"{path}**/*.hdf5"))

    data_list = []

    for ic in icecharts:
        yyyymmdd = ic[-27:-19]
        print(yyyymmdd, end='\r')

        with h5py.File(ic, 'r') as infile:
            sic = infile['y_pred'][:]

        idxs, counts = np.unique(sic, return_counts = True)
    
        data_list.append([yyyymmdd, *counts])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['0', '1', '2', '3', '4', '5', '6'])
    df.to_csv(path_counts)

def read_ml_seasons(path_counts, path):
    icecharts = sorted(glob.glob(f"{path}**/*.hdf5"))

    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"

    with Dataset(f"{path_arome}2019/01/AROME_1kmgrid_20190101T18Z.nc") as constants:
        lsmask = constants['lsmask'][578:,:1792]



    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()

    data_list = []

    for ic in icecharts:
        yyyymmdd = ic[-27:-19]
        print(yyyymmdd, end='\r')

        with h5py.File(ic, 'r') as infile:
            sic = infile['y_pred'][:]

        sic = np.where(lsmask == 1, -1, sic)

        idxs, counts = np.unique(sic, return_counts = True)

    
        data_list.append([f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}", *counts[1:]])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['IFOW', 'OW', 'VODI', 'ODI', 'CDI', 'VCDI', 'FI'])

    for j, idx in zip(range(len(months) - 1), meteorological_seasons):
                df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]

    df.to_csv(path_counts)


def read_amsr2(path_counts, path = None):
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2/2022/"
    
    with Dataset("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/ml/2022/01/weights_21021550_20220105_b20220103.nc", 'r') as nc:
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

        print(f"{idxs=}")
        print(f"{counts=}")
        exit()

        data_list.append([yyyymmdd, *counts[1:]])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['IFOW', 'OW', 'VODI', 'ODI', 'CDI', 'VCDI', 'FI'])

    df.to_csv(path_counts)

def read_amsr2_seasons(path_counts, path = None):
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2/2022/"

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()
    
    with Dataset("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/ml/2022/01/weights_21021550_20220105_b20220103.nc", 'r') as nc:
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

        data_list.append([f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}", *counts[1:]])

    df = pd.DataFrame([i[1:] for i in data_list], index = [i[0] for i in data_list], columns = ['IFOW', 'OW', 'VODI', 'ODI', 'CDI', 'VCDI', 'FI'])

    for j, idx in zip(range(len(months) - 1), meteorological_seasons):
        df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]


    df.to_csv(path_counts)



def main():
    path_counts = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_lead3_singleout_counts_amsr2.csv'
    path_ml = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/weights_30050306/2022/'
    if not os.path.exists(path_counts):
        read_ml_seasons(path_counts, path_ml)

    exit()

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
    ax.set_title('2022 Monthly Sea Ice area fraction ML 3-day lead time', fontsize = 18)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], ['Ice Free Open Water', 'Open Water', 'Very Open Drift Ice', 'Open Drift Ice', 'Close Drift Ice', 'Very Close Drift Ice', 'Fast Ice'][::-1], fontsize = 18)
    plt.savefig('2022-sic-distribution_lead3.pdf')

    

if __name__ == "__main__":
    main()
