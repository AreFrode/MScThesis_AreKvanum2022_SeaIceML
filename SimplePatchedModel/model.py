import os
import tensorflow as tf
from tensorflow.keras import models, layers, losses
from dataset import HDF5Dataset, HDF5Generator

### This should probably be changed to a U-Net or similar, but looks like it works
### (except for the NaN loss appearing after 800 minibatches of the first epoch...)
### The model is training though! :D
### Note that y is currrently flattened in the dataloader, this has to be changed for it to work 
### with a unet architechture.

def test_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, 6, strides=(4, 4), activation='relu', input_shape=(250, 250, 6), data_format='channels_last'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(250 - 6 + 0)/4]+1 = 62
    model.add(layers.Conv2D(32, 6, strides=(4, 4), activation='relu', input_shape=(31, 31, 16), data_format='channels_last'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(62 - 6 + 0)/4]+1 = 15
    model.add(layers.Flatten())
    model.add(tf.keras.Input(shape=(7*7*32)))
    model.add(layers.Dense(62500))

    return model


def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    hdf5data = HDF5Dataset(fname, 'between', seed=14)
    train_data, test_data = hdf5data.homemade_train_test_split()

    BATCH_SIZE = 2

    hdf5generator = HDF5Generator(train_data, fname, batch_size=BATCH_SIZE)

    model = test_model()

    model.summary()

    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True))

    history = model.fit(hdf5generator, epochs=2, batch_size=BATCH_SIZE)





if __name__ == "__main__":
    runstuff()
