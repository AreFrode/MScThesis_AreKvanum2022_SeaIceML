import glob
import h5py
import os

import tensorflow as tf
import numpy as np
from scipy.special import softmax

from unet import create_UNET
from dataset import HDF5Generator


def main():
    SEED_VALUE = 0
    PATH_MODELS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    BATCH_SIZE = 1
    sample_index = 0

    data_2019 = np.array(sorted(glob.glob(f"{PATH_DATA}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{PATH_DATA}2020/**/*.hdf5", recursive=True)))
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5", recursive=True)))

    validation_generator = HDF5Generator(data_2020, batch_size=BATCH_SIZE)


    model = create_UNET(input_shape = (960, 896, 12), channels = [64, 128, 256, 512, 1024])

    
    model.summary()
    load_status = model.load_weights("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/model_2022-29-16_16-29-26").expect_partial()
    
    X, y = validation_generator[sample_index]
    date = validation_generator.get_dates(sample_index)[0][-13:-5]
    
    y_pred = model.predict(X)


    y = np.argmax(softmax(y, axis=-1), axis=-1)
    y_pred = np.argmax(softmax(y_pred, axis=-1), axis=-1)


    output_path = f"{PATH_MODELS}predictions/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = h5py.File(f"{output_path}pred_{sample_index}_16091629.hdf5", 'w-')

    output_file[f"y"] = y
    output_file[f"y_pred"] = y_pred
    output_file[f"date"] = date

    output_file.close()
    



if __name__ == "__main__":
    main()