import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, losses, Input, optimizers
from segmented_dataset import HDF5Dataset, HDF5Generator

def concat_model():
    # Functional API definition
    initializer = keras.initializers.HeNormal()
    inputs1 = Input(shape=(250, 250, 4))
    inputs2 = Input(shape=(500,))

    conv1 = layers.Conv2D(16, 6, strides=(4, 4), activation=tf.nn.relu, input_shape=(250, 250, 4), data_format='channels_last', kernel_initializer=initializer)
    conv2 = layers.Conv2D(32, 6, strides=(4, 4), activation=tf.nn.relu, input_shape=(31, 31, 16), data_format='channels_last', kernel_initializer=initializer)
    relu = layers.Activation(tf.nn.relu)
    flatten = layers.Flatten()

    x1 = layers.Normalization()(inputs1)
    x1 = conv1(x1)  
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = relu(x1)
    x1 = conv2(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = relu(x1)
    x1 = flatten(x1)

    x2 = layers.Normalization()(inputs2)
    x2 = layers.Dense(100, activation=tf.nn.relu, kernel_initializer=initializer)(x2)
    x2 = layers.Dense(10, activation=tf.nn.relu, kernel_initializer=initializer)(x2)

    x = layers.concatenate([x1, x2])

    dense = layers.Dense(2*62500, kernel_initializer=initializer)    
    outputs = dense(x)
    outputs = layers.Reshape((62500,2))(outputs)

    model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)

    return model


def runstuff():
    SEED_VALUE = 0
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)


    hdf5data = HDF5Dataset('between', path_data=PATH_DATA)
    train_data, val_data, test_data = hdf5data.split_by_year

    BATCH_SIZE = 4

    hdf5generator_full = HDF5Generator(train_data, PATH_DATA, n_outputs=2, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)
    hdf5generator_full_test = HDF5Generator(test_data, PATH_DATA, n_outputs=2, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)

    model = concat_model()
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True))

    history = model.fit(hdf5generator_full, epochs=20, batch_size=BATCH_SIZE)
    scores = model.predict(hdf5generator_full_test, batch_size=BATCH_SIZE)

    np.savetxt(f"{PATH_OUTPUT}pred_bins.txt", scores[10], delimiter=',')
    np.savetxt(f"{PATH_OUTPUT}target_bins.txt", hdf5generator_full_test[10][-1][0], delimiter=',')



if __name__ == "__main__":
    runstuff()
