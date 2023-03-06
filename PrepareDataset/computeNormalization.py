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

    data = {}
    for i in range(2016, 2023):
        data[f"{i}"] = (np.array(sorted(glob.glob(f"{path_prepareddata}{i}/**/*.hdf5"))))

    data_train_big = np.concatenate([data[f"{i}"] for i in range(2016, 2021)])
    data_train = np.concatenate((data['2019'], data['2020']))


    fields = ['sic', 
               't2m',
               'xwind',
               'ywind',
               'osisaf_trend_3/sic_trend',
               'osisaf_trend_5/sic_trend',
               'osisaf_trend_7/sic_trend',
               'osisaf_trend_9/sic_trend',
               'osisaf_trend_11/sic_trend',
               'osisaf_trend_31/sic_trend'
               ]
    
    
    for year in range(2019, 2015, -1):
        data_current = np.concatenate([data[f"{i}"] for i in range(year, 2021)])
        compute_and_save_normalization(data_current, fields, path_prepareddata,  f"normalization_constants_train_start_{year}.csv")
    
    # compute_and_save_normalization(data_train_big, fields, path_prepareddata, "normalization_constants_train_big.csv")
    # compute_and_save_normalization(data_train, fields, path_prepareddata, "normalization_constants_train.csv")
    compute_and_save_normalization(data['2021'], fields, path_prepareddata, "normalization_constants_validation.csv")
    compute_and_save_normalization(data['2022'], fields, path_prepareddata, "normalization_constants_test.csv")

if __name__ == "__main__":
    main()
