import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")

import h5py
import os
import glob
import csv

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import UNET, MultiOutputUNET


class HDF5Generator(keras.utils.Sequence):
    def __init__(self, 
                 data, 
                 batch_size = 1, 
                 constant_fields = ['sic', 'sst', 'lsmask'], 
                 dated_fields = ['t2m', 'xwind', 'ywind'], 
                 target = 'sic_target', 
                 num_target_classes = 7, 
                 lower_boundary = 450, 
                 rightmost_boundary = 1840, 
                 normalization_file = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/normalization_constants.csv', 
                 seed=0, 
                 shuffle = True, 
                 augment = True
        ):
        
        self.seed = seed

        self.data = data
        self.rng = np.random.default_rng(self.seed)
        self.batch_size = batch_size
        self.constant_fields = constant_fields
        self.dated_fields = dated_fields
        self.target = target
        self.num_target_classes = num_target_classes
        self.n_fields = len(self.constant_fields) + 2 * len(self.dated_fields)
        self.dim = (2370,1844)  # AROME ARCTIC domain (even numbers)
        self.lower_boundary = lower_boundary
        self.rightmost_boundary = rightmost_boundary

        self.means = {}
        self.stds = {}

        with open(normalization_file, 'r') as f:
            f.readline()

            csv_reader = csv.reader(f, delimiter = ',')
            for row in csv_reader:
                self.means[row[0]] = float(row[1])
                self.stds[row[0]] = float(row[2])

        self.augment = augment
        self.shuffle = shuffle

        self.data_augmentation = keras.Sequential([
            # keras.layers.RandomFlip(mode = 'horizontal'),  # Unsure if random flip is necessary for such a specific domain, also unknown probability for layer triggering
            # keras.layers.RandomRotation(factor = 0.3, fill_mode = 'nearest'),
            # keras.layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1, fill_mode='nearest')
        ])

        if self.shuffle:
            self.rng.shuffle(self.data)

    def __len__(self):
        # Get the number of minibatches
        return int(np.floor(self.data.size / self.batch_size))

    def on_epoch_end(self):
        self.rng.shuffle(self.data)

    def get_dates(self, index):
        return self.data[index*self.batch_size:(index+1)*self.batch_size]

    def get_xy(self, index):
        with h5py.File(self.get_dates(index)[0]) as hf:
            x_vals = hf['x'][:]
            y_vals = hf['y'][:]

        return x_vals[:self.rightmost_boundary], y_vals[self.lower_boundary:]

    def __getitem__(self, index):
        # Get the minibatch associated with index
        samples = self.get_dates(index)
        X, y = self.__generate_data(samples)
        return X, y
        
    def __generate_data(self, samples):
        # Helper function to read data from files
        X = np.empty((self.batch_size, *self.dim, self.n_fields))
        y = np.empty((self.batch_size, *self.dim, self.num_target_classes))

        for idx, sample in enumerate(samples):
            with h5py.File(sample, 'r') as hf:
                X[idx, ..., 0] = hf[f"sic"][:]
                X[idx, ..., 1] = hf[f"sst"][:]
                X[idx, ..., 2] = hf[f"lsmask"][:]
                
                X[idx, ..., 3]   = hf[f"ts0"][f"t2m"][:]
                X[idx, ..., 4]   = hf[f"ts1"][f"t2m"][:]
                X[idx, ..., 5]   = hf[f"ts0"][f"xwind"][:]
                X[idx, ..., 6]   = hf[f"ts1"][f"xwind"][:]
                X[idx, ..., 7]   = hf[f"ts0"][f"ywind"][:]
                X[idx, ..., 8]   = hf[f"ts1"][f"ywind"][:]

                y[idx] = keras.utils.to_categorical(hf[f"{self.target}"][:], num_classes = self.num_target_classes)

        # return X[:, 451::2, :1792:2, :], y[:, 451::2, :1792:2, :]
        return X[:, self.lower_boundary:, :self.rightmost_boundary, :], y[:, self.lower_boundary:, :self.rightmost_boundary, :]

class MultiOutputHDF5Generator(HDF5Generator):
    def __init__(self, 
                 data, 
                 batch_size = 1, 
                 constant_fields = ['sic', 'sst', 'lsmask'], 
                 dated_fields = ['t2m', 'xwind', 'ywind'], 
                 target = 'sic_target', num_target_classes = 7, 
                 lower_boundary = 450, rightmost_boundary = 1840,
                 normalization_file = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/normalization_constants.csv', 
                 seed=0, 
                 shuffle = True, 
                 augment = False
        ):
        HDF5Generator.__init__(self, data, batch_size, constant_fields, dated_fields, target, num_target_classes, lower_boundary, rightmost_boundary, normalization_file, seed, shuffle, augment)

    def __getitem__(self, index):
        # Get the minibatch associated with index
        samples = self.get_dates(index)
        X, y = self.__generate_data(samples)

        if self.augment:
            y = tf.transpose(y, perm = [1, 2, 3, 0])
            X, y = self.__augment_data(X,y)
            y = [y[..., i] for i in range(self.num_target_classes)]

        return X, y

    def __generate_data(self, samples):
        # Helper function to read data from files
        X = np.empty((self.batch_size, *self.dim, self.n_fields))
        y = np.empty((self.batch_size, *self.dim, self.num_target_classes))

        for idx, sample in enumerate(samples):
            with h5py.File(sample, 'r') as hf:
                for i, field in enumerate(self.constant_fields):
                    X[idx, ..., i] = (hf[f"{field}"][:] - self.means[field]) / self.stds[field]

                for j, field in zip(range(len(self.constant_fields), len(self.constant_fields) + 2*len(self.dated_fields), 2), self.dated_fields):
                    X[idx, ..., j] = (hf[f"ts0"][f"{field}"][:] - self.means[f"ts0/{field}"]) / self.stds[f"ts0/{field}"]
                    X[idx, ..., j+1] = (hf[f"ts1"][f"{field}"][:] - self.means[f"ts1/{field}"]) / self.stds[f"ts1/{field}"]

                onehot = hf[f'{self.target}'][:]
                for k in range(self.num_target_classes):
                    y[idx, ..., k] = np.where(onehot >= k, 1, 0)

        return X[:, self.lower_boundary:, :self.rightmost_boundary, :], [y[:, self.lower_boundary:, :self.rightmost_boundary, k] for k in range(self.num_target_classes)]

    def __augment_data(self, X, y):
        concat = tf.concat((X, y), axis = -1)
        concat_aug = self.data_augmentation(concat)
        return concat_aug[..., :self.n_fields], concat_aug[..., self.n_fields:]

    

    
if __name__ == "__main__":
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    data_2019 = np.array(sorted(glob.glob(f"{path_data}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{path_data}2020/**/*.hdf5", recursive=True)))
    data_2021 = np.array(sorted(glob.glob(f"{path_data}2021/**/*.hdf5", recursive=True)))

    train_generator = MultiOutputHDF5Generator(np.concatenate((data_2019, data_2020)), 1, ['sic', 'sst'], lower_boundary=450, rightmost_boundary=1792)
    val_generator = MultiOutputHDF5Generator(data_2021, 1, ['sic', 'sst'], shuffle=False)
    X, y = train_generator[0]
    print(f"{X.shape=}")
    # print(f"{y=}")
    # print(f"{y.shape=}")


    # unet = UNET(channels = [64, 128, 256, 512, 1024])
    unet = MultiOutputUNET(channels = [64, 128, 256, 512], pooling_factor=2)
    y_pred = unet(X)

    print(f"{y_pred[0].shape=}")

