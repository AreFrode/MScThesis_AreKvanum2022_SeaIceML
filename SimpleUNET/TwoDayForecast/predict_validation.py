import sys
# sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/mnt/SimpleUNET")

import glob
import h5py
import os
import csv

import numpy as np
import tensorflow as tf

from unet import create_UNET, create_MultiOutputUNET
from dataset import HDF5Generator, MultiOutputHDF5Generator
from datetime import datetime, timedelta
from helper_functions import read_config_from_csv

def numpy_where_wrapper(arr):
    """Computes the index where the cumulative distribution changes

    Args:
        arr (tensor_like): Cumulative Sea Ice concentration distribution

    Returns:
        int: Highest probable index
    """

    sea_ice_changes = np.where(np.diff(arr))[0]
    changes = len(sea_ice_changes)
        
    # [1. 1. 1. 1. 1. 1. 1.] triggers index error (no change), appropriate class is 6
    # [0. 0. 0. 0. 0. 0. 0.] does as well, should not happen and I assume it does not
    sea_ice_class = sea_ice_changes[0] if changes != 0 else 6
    changes -= 1 if changes != 0 else 0

    return sea_ice_class, changes

def predict_validation_single(validation_generator, model, PATH_OUTPUTS, weights):
    samples = len(validation_generator)

    for i in range(samples):
        print(f"Sample {i} of {samples}", end="\r")
        X, y = validation_generator[i]
        y_pred = np.argmax(model.predict(X), axis=-1)
        
        yyyymmdd = validation_generator.get_dates(i)[0][-13:-5]
        yyyymmdd = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd = (yyyymmdd + timedelta(days = 2)).strftime('%Y%m%d')

        hdf_path = f"{PATH_OUTPUTS}Data/{weights}/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
        if not os.path.exists(hdf_path):
            os.makedirs(hdf_path)

        output_file = h5py.File(f"{hdf_path}SIC_SimpleUNET_two_day_forecast_{yyyymmdd}T15Z.hdf5", "w-")

        output_file["y"] = y
        output_file["y_pred"] = y_pred
        output_file["date"] = yyyymmdd

        output_file.close()

def predict_validation_multi(validation_generator, model, PATH_OUTPUTS, weights):
    samples = len(validation_generator)
    total_changes = 0

    for i in range(samples):
        print(f"Sample {i} of {samples}", end='\r')
        X, y = validation_generator[i]
        y_pred = model.predict(X)
        y_pred = tf.concat(y_pred, axis=-1)
        
        yyyymmdd = validation_generator.get_dates(i)[0][-13:-5]
        yyyymmdd = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd = (yyyymmdd + timedelta(days = 2)).strftime('%Y%m%d')
        
        # out = tf.math.reduce_sum(tf.round(tf.nn.sigmoid(y_pred[0])), -1)
        out = tf.round(tf.nn.sigmoid(y_pred[0]))

        concentration_and_changes = np.apply_along_axis(numpy_where_wrapper, -1, out)

        out = concentration_and_changes[...,0]
        local_changes = concentration_and_changes[...,1]

        hdf_path = f"{PATH_OUTPUTS}Data/{weights}/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
        if not os.path.exists(hdf_path):
            os.makedirs(hdf_path)

        with h5py.File(f"{hdf_path}SIC_SimpleUNET_two_day_forecast_{yyyymmdd}T15Z.hdf5", "w") as output_file:
            output_file["y"] = y
            output_file["y_pred"] = np.expand_dims(out, 0)
            output_file["date"] = yyyymmdd

        total_changes += np.sum(local_changes)

        del X
        del y
        del y_pred
        del concentration_and_changes

    print(f"Total number of inconcistencies for 2021: {total_changes}")

def main():    
    SEED_VALUE = 0
    # PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    # PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    # Rewrite paths for singularity compatibility
    PATH_OUTPUTS = "/mnt/SimpleUNET/TwoDayForecast/outputs/"
    PATH_DATA = "/mnt/PrepareDataset/Data/two_day_forecast/"

    # gpu = tf.config.list_physical_devices('GPU')[0]
    # tf.config.experimental.set_memory_growth(gpu, True)

    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]
    
    # Read config csv
    config = read_config_from_csv(f"{PATH_OUTPUTS}configs/{weights}.csv")

    BATCH_SIZE = 1

    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5")))

    validation_generator = MultiOutputHDF5Generator(data_2021, 
        batch_size=BATCH_SIZE, 
        constant_fields=config['constant_fields'], 
        dated_fields=config['dated_fields'],
        lower_boundary=config['lower_boundary'], 
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['val_normalization']}.csv",
        augment=config['val_augment'],
        shuffle=config['val_shuffle']
    )

    # model = create_UNET(input_shape = (1920, 1840, 9), channels = [64, 128, 256, 512])
    model = create_MultiOutputUNET(
        input_shape = (config['height'], config['width'], len(config['constant_fields']) + 2*len(config['dated_fields'])), 
        channels = config['channels'],
        pooling_factor= config['pooling_factor']
    )

    load_status = model.load_weights(f"{PATH_OUTPUTS}models/{weights}").expect_partial()

    predict_validation_multi(validation_generator, model, PATH_OUTPUTS, weights)




if __name__ == "__main__":
    main()