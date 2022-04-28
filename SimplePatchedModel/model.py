import os
import tensorflow as tf
from tensorflow.keras import models, layers, losses
from dataset import HDF5Dataset, HDF5Generator

### This should probably be changed to a U-Net or similar, but looks like it works
### (except for the NaN loss appearing after ~50% of the first epoch...)
### The model is training though! :D
### Note that y is currrently flattened in the dataloader, this has to be changed for it to work 
### with a unet architechture. 

def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    hdf5data = HDF5Dataset(fname, 'between')
    train_data, test_data = hdf5data.homemade_train_test_split()

    hdf5generator = HDF5Generator(train_data, fname)

    model = models.Sequential()
    model.add(layers.Conv2D(6, 4, strides=(2, 2), activation='relu', input_shape=(250, 250, 6), data_format='channels_last'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(250 - 4 + 0)/2]+1 = 124
    model.add(layers.Conv2D(12, 6, strides=(2, 2), activation='relu', input_shape=(124, 124, 6), data_format='channels_last'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(124 - 6 + 0)/2]+1 = 60
    model.add(layers.Conv2D(12, 8, strides=(2, 2), activation='relu', input_shape=(60, 60, 12), data_format='channels_last'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(60 - 8 + 0)/2]+1 = 27
    model.add(tf.keras.Input(shape=(27*27*18)))
    # model.add(layers.Dense(31250))
    # model.add(layers.Activation('relu'))
    model.add(layers.Dense(62500))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

    history = model.fit_generator(generator = hdf5generator, epochs=2)





if __name__ == "__main__":
    runstuff()
