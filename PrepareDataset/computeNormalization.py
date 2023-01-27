import glob
import os
import h5py
import sys

import numpy as np
import pandas as pd

def compute_and_save_normalization(data, fields, outpath, fname, lower_boundary = 578, eastern_boundary = 1792, domain_side = 1792):
    sinnsykt_stor_array = np.zeros((*data.shape,
                                    domain_side,
                                    domain_side,
                                    len(fields)))
    print(sinnsykt_stor_array.shape)
    
    for i, path in enumerate(data):
        with h5py.File(path, 'r') as f:
            for j, field in enumerate(fields):
                sinnsykt_stor_array[i, :, :, j] = f[field][lower_boundary:, :eastern_boundary]

    means = np.mean(sinnsykt_stor_array, axis = (0,1,2))
    stds = np.std(sinnsykt_stor_array, axis = (0,1,2))
    mins = np.amin(sinnsykt_stor_array, axis = (0,1,2))
    maxs = np.amax(sinnsykt_stor_array, axis = (0,1,2))

    # Append lsmask to stats-lut
    out_fields = fields + ['lsmask']
    means = np.append(means, 0)
    stds = np.append(stds, 1)
    mins = np.append(mins, 0)
    maxs = np.append(maxs, 1)

    out_df = pd.DataFrame({'mean_values': means, 'std_values': stds, 'min_values': mins, 'max_values': maxs}, index=out_fields)
    out_df.to_csv(f"{outpath}{fname}")



def main():
     

    path_prepareddata = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{sys.argv[1]}/"

    data2019 = np.array(sorted(glob.glob(f"{path_prepareddata}2019/**/*.hdf5")))
    data2020 = np.array(sorted(glob.glob(f"{path_prepareddata}2020/**/*.hdf5")))
    data2021 = np.array(sorted(glob.glob(f"{path_prepareddata}2021/**/*.hdf5")))
    data2022 = np.array(sorted(glob.glob(f"{path_prepareddata}2022/**/*.hdf5")))

    data_train = np.concatenate((data2019, data2020))

    fields = ['sic', 
               't2m',
               'xwind',
               'ywind',
               'osisaf_trend_3/sic_trend',
               'osisaf_trend_5/sic_trend',
               'osisaf_trend_7/sic_trend'
               ]

    compute_and_save_normalization(data_train, fields, path_prepareddata, "normalization_constants_train.csv")
    compute_and_save_normalization(data2021, fields, path_prepareddata, "normalization_constants_validation.csv")
    compute_and_save_normalization(data2022, fields, path_prepareddata, "normalization_constants_test.csv")

if __name__ == "__main__":
    main()