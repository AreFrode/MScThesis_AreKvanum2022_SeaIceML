import glob
import h5py
import os
from random import sample
import tensorflow as tf
from tensorflow import keras
import numpy as np

def random_shuffle(hdf5, key):
    names = list(hdf5[key].keys())
    samples = len(names)
    shuffled_names = sample(names, samples)

class HDF5Dataset:
    def __init__(self, fname, key):
        tmpf = h5py.File(fname, 'r')
        self.fname = fname
        self.key = key
        self.temp = 'temp'
        self.dates = list(tmpf[key].keys())
        tmpf.close()
    
    def __call__(self):
        with h5py.File(self.fname, 'r') as hf:
            for _ in range(2):
                date = self.dates[np.random.randint(0, len(self.dates))]
                print(f"{date=}")
                temps = hf[f"{self.key}/{date}/{self.temp}"]
                idx = np.random.randint(0, len(temps[0,:,...]))
                yield temps[:,idx,...]

def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"
    
    if os.path.exists(path_data):
        # files = glob.glob(f"{path_data}*.hdf5")
        fname = f"{path_data}PatchedAromeArctic_201902.hdf5"
        # ds = h5py.File(fname, 'r')

    ds = tf.data.Dataset.from_generator(
        HDF5Dataset(fname, 'between'),
        tf.float16,
        tf.TensorShape([3,250,250])
    )

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