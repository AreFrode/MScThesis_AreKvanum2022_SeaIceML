import glob
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET
from dataset import HDF5Generator


def main():
    SEED_VALUE = 0

    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"
    
    BATCH_SIZE = 1

    data_2019 = np.array(sorted(glob.glob(f"{PATH_DATA}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{PATH_DATA}2020/**/*.hdf5", recursive=True)))

    train_generator = HDF5Generator(np.concatenate((data_2019, data_2020)), batch_size=BATCH_SIZE)
    

    model = create_UNET(input_shape = (960, 896, 6), channels = [64, 128, 256, 512, 1024])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss=keras.losses.CategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

    start = datetime.now()
    model.fit(train_generator, epochs = 20, batch_size = BATCH_SIZE)
    stop = datetime.now()
    print(f"Runtime 20 epochs {stop - start}")




    

if __name__ == "__main__":
    main()