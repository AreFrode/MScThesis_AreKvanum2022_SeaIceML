import glob
import h5py
import os
from random import sample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import numpy as np

# How to deal with 88 / 709 dates only containing a single entry?
# How to deal with other thresholds than between to build a balanced dataset?
#   Maybe create three datasets, one for each threshold, then merge them together 
#   Into one larger dataset. Using the approach I am outlining, placement would 
#   not matter.

class HDF5Dataset:
    def __init__(self, fname, key, seed=0):
        self.seed = seed
        np.random.seed(self.seed)
        tmpf = h5py.File(fname, 'r')
        self.fname = fname
        self.key = key
        self.fields = ['xc', 'yc', 'sst', 't2m', 'xwind', 'ywind', 'sic']
        self.dates = list(tmpf[key].keys())
        self.data = []
        for date in self.dates:
            self.data.append({date: np.arange(0, tmpf[key][date][self.fields[0]].shape[0])})

        tmpf.close()

    def homemade_train_test_split(self, train_size = 0.75):
        # This gives some dates a much higher probability of being in the training data 
        # since patches from dates containing fewer patches are more likely too get chosen
        # I want some input on how to better split, or even construct the dataset

        data_copy = self.data.copy()
        self.train_data = np.empty((len(data_copy),), object)

        for i in range(self.train_data.shape[0]):
            self.train_data[i] = {self.dates[i]: []}
        
        elements = count_elements(data_copy)

        current = 0
        target = int(train_size*elements)

        while current < target:
            date = np.random.randint(0, len(self.dates))

            if len(list(data_copy[date].values())[0]) == 0:
               continue

            patch = np.random.choice(list(data_copy[date].values())[0])
            data_copy[date] = {self.dates[date] : np.delete(list(data_copy[date].values())[0], np.where(list(data_copy[date].values())[0] == patch))}
            self.train_data[date][list(self.train_data[date])[0]].append(patch)

            current = count_elements(self.train_data)

        train_remove_idx_list = []
        for idx, date in enumerate(self.train_data):
            if len(list(date.values())[0]) == 0:
                train_remove_idx_list.append(idx)

            list(date.values())[0].sort()

        self.train_data = np.delete(self.train_data, train_remove_idx_list)
        
        self.test_data = []
        for date in data_copy:
            if list(date.values())[0].size != 0:
                self.test_data.append(date)

        self.test_data = np.array(self.test_data)

        return self.train_data, self.test_data

def count_elements(data):
    elements = 0
    for date in data:
        elements += len(list(date.values())[0])

    return elements


class HDF5Generator(Sequence):
    # For later, what about shuffling the rekkefølge of the data for every epoch?

    def __init__(self, data, fname, key='between', batch_size=1, dim=(250,250), n_fields=6):
        self.data = data
        self.key = key
        self.dim = dim
        self.n_fields = n_fields
        self.batch_size = batch_size
        self.fname = fname
        self.fields = ['xc', 'yc', 'sst', 't2m', 'xwind', 'ywind', 'sic']

    def __len__(self):
        # Get the number of minibatches
        return int(np.floor(count_elements(self.data)) / self.batch_size)

    def __getitem__(self, index):
        # Get the minibatch associated with index
        samples = []
        indexes = np.random.choice(self.data, size=(self.batch_size))
        
        for sample in indexes:
            date = list(sample.keys())[0]
            patch = np.random.choice(list(sample.values())[0])
            samples.append({date:patch})

        X, y = self.__generate_data(samples)
        return X, y

    def __generate_data(self, samples):
        # Helper function 

        X = np.empty((self.batch_size, *self.dim, self.n_fields))
        y = np.empty((self.batch_size, np.prod(self.dim)))

        with h5py.File(self.fname, 'r') as hf:
            for idx, sample in enumerate(samples):
                date = list(sample.keys())[0]
                patch = list(sample.values())[0]

                xc = hf[f"{self.key}/{date}/{self.fields[0]}"][patch, ...]
                yc = hf[f"{self.key}/{date}/{self.fields[1]}"][patch, ...]
                sst = hf[f"{self.key}/{date}/{self.fields[2]}"][0, patch, ...]
                t2m = hf[f"{self.key}/{date}/{self.fields[3]}"][0, patch, ...]
                xwind = hf[f"{self.key}/{date}/{self.fields[4]}"][0, patch, ...]
                ywind = hf[f"{self.key}/{date}/{self.fields[5]}"][0, patch, ...]
                sic = hf[f"{self.key}/{date}/{self.fields[6]}"][patch, ...]
                
                out_x = np.stack((xc, yc, sst, t2m, xwind, ywind), axis=-1)

                X[idx, ...] = out_x
                y[idx, ...] = sic.flatten()
    
        return X, y



def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"
    
    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    
    generator = HDF5Dataset(fname, 'between')

    train_data, test_data = generator.homemade_train_test_split()
    
    hdf5generator = HDF5Generator(train_data, fname)

    hdf5generator[0]

    



if __name__ == "__main__":
    runstuff()