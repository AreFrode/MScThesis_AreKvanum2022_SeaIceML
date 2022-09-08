import h5py
import os
import glob

import numpy as np

from tensorflow import keras


class HDF5Generator(keras.utils.Sequence):
    def __init__(self, data, batch_size, key= 'day0', constant_fields = ['sst', 'lsmask'], dated_fields = ['t2m', 'xwind', 'ywind'], target = 'sic_target', num_target_classes = 6, seed=0):
        self.seed = seed

        self.data = data
        self.rng = np.random.default_rng(self.seed)
        self.batch_size = batch_size
        self.key = key
        self.constant_fields = constant_fields
        self.dated_fields = dated_fields
        self.target = target
        self.num_target_classes = num_target_classes
        self.n_fields = len(self.constant_fields) + len(self.dated_fields)
        self.dim = (948,738)  # AROME ARCTIC domain (even numbers)

        self.rng.shuffle(self.data)

    def __len__(self):
        # Get the number of minibatches
        return int(np.floor(self.data.size / self.batch_size))

    def on_epoch_end(self):
        self.rng.shuffle(self.data)

    def __getitem__(self, index):
        # Get the minibatch associated with index
        samples = self.data[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__generate_data(samples)
        return X, y
        
    def __generate_data(self, samples):
        # Helper function to read data from files
        X = np.empty((self.batch_size, *self.dim, self.n_fields))
        y = np.empty((self.batch_size, *self.dim, self.num_target_classes))

        for idx, sample in enumerate(samples):
            with h5py.File(sample, 'r') as hf:
                for i, field in enumerate(self.constant_fields):
                    X[idx, ..., i] = hf[f"{field}"][:]

                for j, field in enumerate(self.dated_fields, start = i+1):
                    X[idx, ..., j] = hf[f"{self.key}"][f"{field}"][:]

                y[idx] = keras.utils.to_categorical(hf[f"{self.target}"][:])

        return X, y




if __name__ == "__main__":
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    data_2019 = np.array(sorted(glob.glob(f"{path_data}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{path_data}2020/**/*.hdf5", recursive=True)))
    data_2021 = np.array(sorted(glob.glob(f"{path_data}2021/**/*.hdf5", recursive=True)))
    
    train_generator = HDF5Generator(data_2019, 1)
    X, y = train_generator[0]