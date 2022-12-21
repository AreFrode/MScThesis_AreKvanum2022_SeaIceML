import glob
import os
import h5py

import numpy as np
import pandas as pd

from tqdm import tqdm

def compute_and_save_normalization(data, fields, outpath, fname, lower_boundary = 578, western_boundary = 1792, domain_side = 1792):
    sinnsykt_stor_array = np.zeros((*data.shape,
                                    domain_side,
                                    domain_side,
                                    len(fields)))
    print(sinnsykt_stor_array.shape)
    
    for i, path in enumerate(tqdm(data)):
        with h5py.File(path, 'r') as f:
            for j, field in enumerate(fields):
                sinnsykt_stor_array[i, :, :, j] = f[field][lower_boundary:, :western_boundary]

    means = np.mean(sinnsykt_stor_array, axis = (0,1,2))
    stds = np.std(sinnsykt_stor_array, axis = (0,1,2))

    out_df = pd.DataFrame({'mean_values': means, 'std_values': stds}, index=fields)
    out_df.to_csv(f"{outpath}{fname}")



def main():
    path_prepareddata = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    data2019 = np.array(sorted(glob.glob(f"{path_prepareddata}2019/**/*.hdf5")))
    data2020 = np.array(sorted(glob.glob(f"{path_prepareddata}2020/**/*.hdf5")))
    data2021 = np.array(sorted(glob.glob(f"{path_prepareddata}2021/**/*.hdf5")))

    data_train = np.concatenate((data2019, data2020))

    fields = ['sic', 
               'sic_trend', 
               'ts0/t2m',
               'ts1/t2m',
               'ts0/xwind',
               'ts1/xwind',
               'ts0/ywind',
               'ts1/ywind']

    compute_and_save_normalization(data_train, fields, path_prepareddata, "normalization_constants_train.csv")
    compute_and_save_normalization(data2021, fields, path_prepareddata, "normalization_constants_validation.csv")

if __name__ == "__main__":
    main()