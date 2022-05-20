import glob
import h5py
import os
from random import sample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import numpy as np

# The purpose of this dataset class is to be able to handle several smaller hdf5 files
# Structured by year and month (as MET data usually is) compared to one large file
# Other than that, the file should function identically
# Note that not only are the files differently structured, the SIC is onehot encoded to either 
# contain ice or no ice.

# PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

class HDF5Dataset:
    def __init__(self, key, path_data):
        # tmpf = h5py.File(fname, 'r')
        # self.fname = fname
        self.key = key
        self.path_data = path_data
        self.fields = ['xc', 'yc', 'sst', 't2m', 'xwind', 'ywind', 'sic']
        self.dates = glob.glob(f"{self.path_data}20*/*/*.hdf5", recursive=True)
        for i in range(len(self.dates)):
            self.dates[i] = self.dates[i][-13:-5]

        self.dates = sorted(self.dates)

        self.data = []
        for date in self.dates:
            with h5py.File(f"{self.path_data}{date[:4]}/{date[4:6]}/PatchedAromeArcticBins_{date}.hdf5") as tmpf:
                self.data.append({date: np.arange(0, tmpf[key][self.fields[0]].shape[0])})

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
    def __init__(self, data, path_data, key='between', batch_size=1, dim=(250,250), n_fields=4, n_outputs=1, SEED_VALUE=0):
        self.seed = SEED_VALUE
        np.random.seed(self.seed)

        self.data = data
        np.random.shuffle(self.data)

        self.path_data = path_data
        self.key = key
        self.dim = dim
        self.n_fields = n_fields
        self.batch_size = batch_size
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

        y = np.empty((self.batch_size, np.prod(self.dim), 2))

        for idx, sample in enumerate(samples):
            date = list(sample.keys())[0]
            patch = list(sample.values())[0]
            with h5py.File(f"{self.path_data}{date[:4]}/{date[4:6]}/PatchedAromeArcticBins_{date}.hdf5", 'r') as hf:
                xc = hf[f"{self.key}/{self.fields[0]}"][patch, 0, :]
                yc = hf[f"{self.key}/{self.fields[1]}"][patch, :, 0]

                sst = hf[f"{self.key}/{self.fields[2]}"][0, patch, ...]
                t2m = hf[f"{self.key}/{self.fields[3]}"][0, patch, ...]
                xwind = hf[f"{self.key}/{self.fields[4]}"][0, patch, ...]
                ywind = hf[f"{self.key}/{self.fields[5]}"][0, patch, ...]
                sic = hf[f"{self.key}/{self.fields[6]}"][patch, ...]
                
                out_x1 = np.stack((sst, t2m, xwind, ywind), axis=-1)
                out_x2 = np.stack((xc, yc), axis=-1).flatten()

                X1[idx, ...] = out_x1
                X2[idx, ...] = out_x2

                X = [X1, X2]
                y[idx] = sic
                

        return X[:self.n_outputs], y



def runstuff():
    # print(f"{tf.config.list_physical_devices()=}")

    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"
    
    generator = HDF5Dataset(key='between', path_data=path_data)

    # generator.split_by_year

    train_data, val_data, test_data = generator.split_by_year
    
    hdf5generator = HDF5Generator(train_data, path_data=path_data)

    print(f"{hdf5generator[0]=}")

    



if __name__ == "__main__":
    runstuff()