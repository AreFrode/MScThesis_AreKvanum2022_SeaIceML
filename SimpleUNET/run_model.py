import glob
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET
from dataset import HDF5Generator


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

    if not os.path.exists(f"{PATH_OUTPUT}models"):
        os.makedirs(f"{PATH_OUTPUT}models")

    train_generator = HDF5Generator(np.concatenate((data_2019, data_2020)), batch_size=BATCH_SIZE)
    
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = 1000000,
        decay_rate = .96,
        staircase=True
    )

    model = create_UNET(input_shape = (960, 896, 6), channels = [64, 128, 256, 512, 1024])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss=keras.losses.CategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

    model.summary()
    # history = model.fit(train_generator, validation_data = validation_generator, epochs = 1, batch_size = BATCH_SIZE)
    history = model.fit(train_generator, epochs = 20, batch_size = BATCH_SIZE)

    current_time = datetime.now().strftime("%d%m%H%M")

    print(f"Current time when saving model {current_time}")
    model.save_weights(f'{PATH_OUTPUT}models/weights_{current_time}', save_format='tf')
    # model.save_weights(f'{PATH_OUTPUT}test_model', save_format='tf')
    

if __name__ == "__main__":
    main()