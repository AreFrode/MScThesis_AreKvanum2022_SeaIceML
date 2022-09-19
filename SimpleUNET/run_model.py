import glob
import os
import h5py
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET, UNET
from dataset import HDF5Generator
from scipy.special import softmax


def main():
    # print(tf.config.list_physical_devices('GPU'))

    SEED_VALUE = 0
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"
    
    BATCH_SIZE = 1

    data_2019 = np.array(sorted(glob.glob(f"{PATH_DATA}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{PATH_DATA}2020/**/*.hdf5", recursive=True)))
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5", recursive=True)))

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    train_generator = HDF5Generator(data_2019, batch_size=BATCH_SIZE)
    validation_generator = HDF5Generator(data_2020, batch_size=BATCH_SIZE)
    
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = 1000000,
        decay_rate = .96,
        staircase=True
    )

    model = create_UNET(input_shape = (960, 896, 12), channels = [64, 128, 256, 512, 1024])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss=keras.losses.CategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

    model.summary()
    # history = model.fit(train_generator, validation_data = validation_generator, epochs = 1, batch_size = BATCH_SIZE)
    history = model.fit(train_generator, epochs = 20, batch_size = BATCH_SIZE)
    
    model.save_weights(f'{PATH_OUTPUT}model_{datetime.now().strftime("%d%m%H%M")}', save_format='tf')
    # model.save_weights(f'{PATH_OUTPUT}test_model', save_format='tf')

    X, y = validation_generator[0]
    # y = np.argmax(keras.activations.softmax(y))
    y_pred = np.argmax(model.predict(X), axis=-1)

    prediction_path = f"{PATH_OUTPUT}predictions/"
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    output_file = h5py.File(f"{prediction_path}pred_runmodel{datetime.now().strftime('%d%m%H%M')}.hdf5", 'w-')

    # output_file[f"y"] = y
    output_file[f"y_pred"] = y_pred

    output_file.close()
    


if __name__ == "__main__":
    main()