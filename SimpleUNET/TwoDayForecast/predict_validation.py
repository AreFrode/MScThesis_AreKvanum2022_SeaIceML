import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")

import glob
import h5py
import os

import numpy as np
import tensorflow as tf

from unet import create_UNET, create_MultiOutputUNET
from dataset import HDF5Generator, MultiOutputHDF5Generator
from datetime import datetime, timedelta

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
        print(f"Sample {i} of {samples}", end="\r")
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

    print(f"Total number of inconcistencies for 2021: {total_changes}")

def main():
    SEED_VALUE = 0
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    BATCH_SIZE = 1
    constant_fields = ['sic', 'sic_trend', 'lsmask']
    dated_fields = ['t2m', 'xwind', 'ywind']


    weights = "weights_01111912"
    
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5", recursive=True)))

    # validation_generator = HDF5Generator(data_2021, batch_size=BATCH_SIZE, shuffle=False)
    validation_generator = MultiOutputHDF5Generator(data_2021, 
        batch_size=BATCH_SIZE, 
        constant_fields=constant_fields, 
        dated_fields=dated_fields, 
        lower_boundary=578, 
        rightmost_boundary=1792, 
        augment = False,
        shuffle=False
    )

    # model = create_UNET(input_shape = (1920, 1840, 9), channels = [64, 128, 256, 512])
    model = create_MultiOutputUNET(input_shape = (1792, 1792, len(constant_fields) + 2*len(dated_fields)), channels = [64, 128, 256, 512], pooling_factor=4)

    load_status = model.load_weights(f"{PATH_OUTPUTS}models/{weights}").expect_partial()

    predict_validation_multi(validation_generator, model, PATH_OUTPUTS, weights)




if __name__ == "__main__":
    main()