import os
import numpy as np
from dataset import HDF5Dataset, HDF5Generator


def runstuff():
    DATA_PATH = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    year = 2021
    month = [i for i in range(1, 13)]
    
    if os.path.exists(DATA_PATH):
        fname = f"{DATA_PATH}FullPatchedAromeArctic.hdf5"

    hdf5data = HDF5Dataset(fname, 'between', seed=0)
    train_data, val_data, test_data = hdf5data.split_by_year

    hdf5generator_single_test = HDF5Generator(test_data, fname, batch_size=1, SEED_VALUE=1)

    for i in range(len(hdf5generator_single_test)):
        print(i)
        X, y = hdf5generator_single_test[i]

        if np.ma.is_masked(X) or np.ma.is_masked(y):
            print(f"Bad batch")




if __name__ == "__main__":
    runstuff()