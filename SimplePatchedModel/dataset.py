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

    @property
    def split_by_year(self):
        """Splits the data in train/val/test subsets,
            train covers 2019, 248 correct end 2019
            val covers 2020, 467 correct end 2020
            test covers 2021, -1 correct End 2021
        """

        self.train = self.extract_and_isolate_data(stop=249)
        self.val = self.extract_and_isolate_data(start=249,stop=468)
        self.test = self.extract_and_isolate_data(start=468)

        return self.train, self.val, self.test
    
    def extract_and_isolate_data(self, start = 0, stop = -1):
        """Extract the correct date and patches, as well as isolates each datapoint as seperate array entry

        Args:
            start (int, optional): index to start slicing the data array. Defaults to 0.
            stop (int, optional): index to stop slicing the data array. Defaults to -1.

        Returns:
            np.array(object): data subset with each patch as separate entry
        """

        data = np.empty((count_elements(self.data[start:stop]),), object)
        displacement = 0

        for i, date in enumerate(self.dates[start:stop]):
            for value in list(self.data[i + start].values())[0]:
                data[displacement] = {date: value}
                displacement += 1

        return data

def count_elements(data):
    elements = 0
    for date in data:
        elements += len(list(date.values())[0])

    return elements


class HDF5Generator(Sequence):
    # For later, what about shuffling the rekkefølge of the data for every epoch?

    def __init__(self, data, fname, key='between', batch_size=1, dim=(250,250), n_fields=4, n_outputs=1, SEED_VALUE=0):
        self.seed = SEED_VALUE
        np.random.seed(self.seed)

        self.data = data
        np.random.shuffle(self.data)

        self.key = key
        self.dim = dim
        self.n_fields = n_fields
        self.batch_size = batch_size
        self.fname = fname
        self.n_outputs = n_outputs
        self.fields = ['xc', 'yc', 'sst', 't2m', 'xwind', 'ywind', 'sic']

    def __len__(self):
        # Get the number of minibatches
        return int(np.floor(self.data.size / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.data)

    def __getitem__(self, index):
        # Get the minibatch associated with index
        # print(f"{self.data[index]=}")
        samples = self.data[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__generate_data(samples)
        return X, y

    def __generate_data(self, samples):
        # Helper function 

        X1 = np.empty((self.batch_size, *self.dim, self.n_fields))
        X2 = np.empty((self.batch_size, 2*self.dim[0]))

        y = np.empty((self.batch_size, np.prod(self.dim)))

        with h5py.File(self.fname, 'r') as hf:
            for idx, sample in enumerate(samples):
                date = list(sample.keys())[0]
                patch = list(sample.values())[0]

                xc = hf[f"{self.key}/{date}/{self.fields[0]}"][patch, 0, :]
                yc = hf[f"{self.key}/{date}/{self.fields[1]}"][patch, :, 0]

                sst = hf[f"{self.key}/{date}/{self.fields[2]}"][0, patch, ...]
                t2m = hf[f"{self.key}/{date}/{self.fields[3]}"][0, patch, ...]
                xwind = hf[f"{self.key}/{date}/{self.fields[4]}"][0, patch, ...]
                ywind = hf[f"{self.key}/{date}/{self.fields[5]}"][0, patch, ...]
                sic = hf[f"{self.key}/{date}/{self.fields[6]}"][patch, ...]
                
                out_x1 = np.stack((sst, t2m, xwind, ywind), axis=-1)
                out_x2 = np.stack((xc, yc), axis=-1).flatten()

                X1[idx, ...] = out_x1
                X2[idx, ...] = out_x2

                X = [X1, X2]
                y[idx] = np.where(sic.flatten() >= 0.5, 1, 0)
                

        return X[:self.n_outputs], y



def runstuff():
    # print(f"{tf.config.list_physical_devices()=}")


    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"
    
    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    
    generator = HDF5Dataset(fname, 'between')

    # generator.split_by_year

    train_data, val_data, test_data = generator.split_by_year
    
    hdf5generator = HDF5Generator(train_data, fname)

    hdf5generator[0]

    



if __name__ == "__main__":
    runstuff()