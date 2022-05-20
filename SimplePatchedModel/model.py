import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, losses, Input, optimizers
from dataset import HDF5Dataset, HDF5Generator


"""
The encoder model seems to be very sensitive to exploding gradients
This is a problem, as the model can break either during training or inference
"""

def encoder():
    inputs = Input(shape=(250, 250, 4))

    conv1 = layers.Conv2D(16, 6, strides=(4, 4), activation='relu', input_shape=(250, 250, 4), data_format='channels_last')
    conv2 = layers.Conv2D(32, 6, strides=(4, 4), activation='relu', input_shape=(31, 31, 16), data_format='channels_last')
    relu = layers.Activation(tf.nn.leaky_relu)
    flatten = layers.Flatten()
    dropout = layers.Dropout(.1)
    x = dropout(inputs)
    x = conv1(x)  
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = dropout(x)
    x = conv2(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = flatten(x)

    dense = layers.Dense(62500)    
    outputs = dense(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def concat_model():
    # Functional API definition
    initializer = keras.initializers.HeNormal()
    inputs1 = Input(shape=(250, 250, 4))
    inputs2 = Input(shape=(500,))

    conv1 = layers.Conv2D(16, 6, strides=(4, 4), activation=tf.nn.relu, input_shape=(250, 250, 4), data_format='channels_last', kernel_initializer=initializer)
    conv2 = layers.Conv2D(32, 6, strides=(4, 4), activation=tf.nn.relu, input_shape=(31, 31, 16), data_format='channels_last', kernel_initializer=initializer)
    relu = layers.Activation(tf.nn.relu)
    flatten = layers.Flatten()

    x1 = conv1(inputs1)  
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = relu(x1)
    x1 = conv2(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = relu(x1)
    x1 = flatten(x1)

    x2 = layers.Dense(1000, activation=tf.nn.relu, kernel_initializer=initializer)(inputs2)
    x2 = layers.Dense(100, activation=tf.nn.relu, kernel_initializer=initializer)(x2)

    x = layers.concatenate([x1, x2])

    dense = layers.Dense(62500, kernel_initializer=initializer)    
    outputs = dense(x)

    model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)

    return model


def runstuff():
    SEED_VALUE = 0
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimplePatchedModel/outputs/"
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    if os.path.exists(path_data):
        fname = f"{path_data}FullPatchedAromeArctic.hdf5"

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)


    hdf5data = HDF5Dataset(fname, 'between', seed=0)
    train_data, val_data, test_data = hdf5data.split_by_year

    BATCH_SIZE = 4

    hdf5generator_single = HDF5Generator(train_data, fname, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)

    hdf5generator_single_test = HDF5Generator(test_data, fname, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)

    hdf5generator_full = HDF5Generator(train_data, fname, n_outputs=2, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)
    hdf5generator_full_test = HDF5Generator(test_data, fname, n_outputs=2, batch_size=BATCH_SIZE, SEED_VALUE=SEED_VALUE)

    """
    model1 = encoder()
    model1.summary()
    model1.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.BinaryCrossentropy(from_logits=True))
    history = model1.fit(hdf5generator_single, epochs=20, batch_size=BATCH_SIZE)
    scores = model1.evaluate(hdf5generator_single_test)
    """

    model2 = concat_model()
    model2.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True))
    history = model2.fit(hdf5generator_full, epochs=15, batch_size=BATCH_SIZE)
    scores = model2.predict(hdf5generator_full_test, batch_size=BATCH_SIZE)

    np.savetxt(f"{PATH_OUTPUT}pred.txt", scores[0], delimiter=',')
    np.savetxt(f"{PATH_OUTPUT}target.txt", hdf5generator_full_test[0][-1][0], delimiter=',')



if __name__ == "__main__":
    runstuff()
