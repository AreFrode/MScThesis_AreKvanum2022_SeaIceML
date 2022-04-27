import glob
import h5py
import os
from random import sample
import tensorflow as tf
from tensorflow import keras
import numpy as np

# How to deal with 88 / 709 dates only containing a single entry?
# How to deal with other thresholds than between to build a balanced dataset?
#   Maybe create three datasets, one for each threshold, then merge them together 
#   Into one larger dataset. Using the approach I am outlining, placement would 
#   not matter.

def random_shuffle(hdf5, key):
    names = list(hdf5[key].keys())
    samples = len(names)
    shuffled_names = sample(names, samples)

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
        
        elements = self._count_elements(data_copy)

        current = 0
        target = int(train_size*elements)

        while current < target:
            date = np.random.randint(0, len(self.dates))

            if len(list(data_copy[date].values())[0]) == 0:
               continue

            patch = np.random.choice(list(data_copy[date].values())[0])
            data_copy[date] = {self.dates[date] : np.delete(list(data_copy[date].values())[0], np.where(list(data_copy[date].values())[0] == patch))}
            self.train_data[date][list(self.train_data[date])[0]].append(patch)

            current = self._count_elements(self.train_data)

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

          
    def _count_elements(self, data):
        elements = 0
        for date in data:
            elements += len(list(date.values())[0])

        return elements

    def __call__(self):
        with h5py.File(self.fname, 'r') as hf:
            for _ in range(2):
                sample = np.random.choice(self.train_data)
                date = list(sample.keys())[0]
                patch = np.random.choice(list(sample.values())[0])

                xc = hf[f"{self.key}/{date}/{self.fields[0]}"]
                xc = xc[patch, ...]
                xc = np.repeat(xc[np.newaxis,...], 3, axis=0)

                yc = hf[f"{self.key}/{date}/{self.fields[1]}"]
                yc = yc[patch, ...]
                yc = np.repeat(yc[np.newaxis,...], 3, axis=0)

                sst = hf[f"{self.key}/{date}/{self.fields[2]}"]
                t2m = hf[f"{self.key}/{date}/{self.fields[3]}"]
                xwind = hf[f"{self.key}/{date}/{self.fields[4]}"]
                ywind = hf[f"{self.key}/{date}/{self.fields[5]}"]

                sic = hf[f"{self.key}/{date}/{self.fields[6]}"]
                sic = sic[patch, ...]
                sic = np.repeat(sic[np.newaxis,...], 3, axis=0)

                out = np.stack((xc, yc, sst[:,patch,...], t2m[:,patch,...], xwind[:,patch,...], ywind[:,patch,...], sic))
                
                yield out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"
    
    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    generator = HDF5Dataset(fname, 'between')

    ds = tf.data.Dataset.from_generator(
        generator,
        tf.float32,
        tf.TensorShape([7,3,250,250])
    )

    generator.homemade_train_test_split()

    it = iter(ds)

    while True:
        try:
            print(it.get_next())

        except tf.errors.OutOfRangeError:
            print('Sampled through all batches')
            break
    

    
    # random_shuffle(ds, 'between')



if __name__ == "__main__":
    runstuff()