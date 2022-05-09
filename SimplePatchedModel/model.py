import os
import tensorflow as tf
from tensorflow.keras import models, layers, losses, Input
from dataset import HDF5Dataset, HDF5Generator

### This should probably be changed to a U-Net or similar, but looks like it works
### (except for the NaN loss appearing after 800 minibatches of the first epoch...)
### The model is training though! :D
### Note that y is currrently flattened in the dataloader, this has to be changed for it to work 
### with a unet architechture.

def test_model():
    inputs = Input(shape=(250, 250, 4))
    inputs2 = Input(shape=(7*7*32))
    conv1 = layers.Conv2D(16, 6, strides=(4, 4), activation='relu', input_shape=(250, 250, 4), data_format='channels_last')
    conv2 = layers.Conv2D(32, 6, strides=(4, 4), activation='relu', input_shape=(31, 31, 16), data_format='channels_last')
    bn = layers.BatchNormalization()
    pooling = layers.MaxPooling2D(pool_size=(2, 2))
    relu = layers.Activation('relu')
    flatten = layers.Flatten()
    dense = layers.Dense(62500)


    x = conv1(inputs)
    # x = bn(x)
    x = pooling(x)
    x = relu(x)
    x = conv2(x)
    # x = bn(x)
    x = pooling(x)
    x = relu(x)
    x = flatten(x)
    outputs = dense(x)

    model1 = models.Model(inputs=inputs, outputs=outputs)
    """
    model1 = models.Sequential()
    model1.add(layers.Conv2D(16, 6, strides=(4, 4), activation='relu', input_shape=(250, 250, 4), data_format='channels_last'))
    model1.add(layers.BatchNormalization())
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model1.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(250 - 6 + 0)/4]+1 = 62
    model1.add(layers.Conv2D(32, 6, strides=(4, 4), activation='relu', input_shape=(31, 31, 16), data_format='channels_last'))
    model1.add(layers.BatchNormalization())
    model1.add(layers.MaxPool2D(pool_size=(2, 2)))
    model1.add(layers.Activation('relu'))
    # [(W - k + 2P)/S]+1, [(62 - 6 + 0)/4]+1 = 15
    model1.add(layers.Flatten())
    """

    """
    model2 = models.Sequential()
    model2.add(layers.Flatten())
    model2.add(Input(shape=(500)))
    model2.add(layers.Dense(200))

    model_concat = layers.concatenate([model1.output, model2.output], axis=-1)

    model_concat = layers.Dense(62500)(model_concat)
    # model1.add(Input(shape=(7*7*32)))
    # model_concat.add(layers.Dense(62500))

    model = models.Model(inputs=[model1.input, model2.input], outputs=model_concat)
    """

    return model1


def runstuff():
    SEED_VALUE = 0
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    hdf5data = HDF5Dataset(fname, 'between', seed=0)
    train_data, val_data, test_data = hdf5data.split_by_year

    BATCH_SIZE = 2

    hdf5generator = HDF5Generator(train_data, fname, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)
    hdf5generator[0]
    model = test_model()

    model.summary()

    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True))

    history = model.fit(hdf5generator, epochs=2, batch_size=BATCH_SIZE)





if __name__ == "__main__":
    runstuff()
