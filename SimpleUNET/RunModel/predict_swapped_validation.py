import sys
# sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/mnt/SimpleUNET")

import glob
import h5py
import os
import csv

import numpy as np
import tensorflow as tf

from unet import create_MultiOutputUNET
from dataset import SwappedMultiOutputHDF5Generator
from datetime import datetime, timedelta
from helper_functions import read_config_from_csv
from predict_validation import numpy_where_wrapper


def predict_test_multi(test_generator, model, PATH_OUTPUTS, weights, lead_time, name, seed):
    samples = len(test_generator)
    total_changes = 0

    for i in range(samples):
        print(f"Sample {i} of {samples}", end='\r')
        X, y = test_generator[i]

        y_pred = model.predict(X)
        y_pred = tf.concat(y_pred, axis=-1)
        
        yyyymmdd = test_generator.get_dates(i)[0][-13:-5]
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d')
        
        # out = tf.math.reduce_sum(tf.round(tf.nn.sigmoid(y_pred[0])), -1)
        out = tf.round(tf.nn.sigmoid(y_pred[0]))

        concentration_and_changes = np.apply_along_axis(numpy_where_wrapper, -1, out)

        out = concentration_and_changes[...,0]
        local_changes = concentration_and_changes[...,1]

        x_vals, y_vals = test_generator.get_xy(i)

        hdf_path = f"{PATH_OUTPUTS}Data/swapped/seed_{seed}/{weights}/overwritten_{name}/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
        if not os.path.exists(hdf_path):
            os.makedirs(hdf_path)

        with h5py.File(f"{hdf_path}SIC_UNET_v{yyyymmdd_valid}_b{yyyymmdd}T15Z.hdf5", "w") as output_file:
            output_file['xc'] = x_vals
            output_file['yc'] = y_vals
            output_file["y"] = y
            output_file["y_pred"] = np.expand_dims(out, 0)
            output_file["date"] = yyyymmdd

        total_changes += np.sum(local_changes)

        del X
        del y
        del y_pred
        del concentration_and_changes

    print(f"Total number of inconcistencies for 2022: {total_changes}")

def main():    
    # PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    # PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]
    seed_value = int(sys.argv[2])

    
    # Read config csv
    PATH_OUTPUTS = "/mnt/SimpleUNET/RunModel/outputs/"
    config = read_config_from_csv(f"{PATH_OUTPUTS}configs/{weights}.csv")

    # Rewrite paths for singularity compatibility

    if config['open_ocean_mask']:
        PATH_DATA = f"/mnt/PrepareDataset/Data/open_ocean/lead_time_{config['lead_time']}/"

    elif config['reduced_classes']:
        PATH_DATA = f"/mnt/PrepareDataset/Data/reduced_classes/lead_time_{config['lead_time']}/"

    else:
        PATH_DATA = f"/mnt/PrepareDataset/Data/lead_time_{config['lead_time']}/"

    # gpu = tf.config.list_physical_devices('GPU')[0]
    # tf.config.experimental.set_memory_growth(gpu, True)

    BATCH_SIZE = 1

    data_2022 = np.array(sorted(glob.glob(f"{PATH_DATA}2022/**/*.hdf5")))

    # model = create_UNET(input_shape = (1920, 1840, 9), channels = [64, 128, 256, 512])
    model = create_MultiOutputUNET(
        input_shape = (config['height'], config['width'], len(config['fields'])), 
        channels = config['channels'],
        pooling_factor = config['pooling_factor'],
        num_outputs = config['num_outputs'],
        average_pool = config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

    load_status = model.load_weights(f"{PATH_OUTPUTS}models/{weights}").expect_partial()

    overwritten = ['sic', 'trend', 'lsmask', 't2m', 'xwind', 'ywind']

    for i, name in zip(range(len(overwritten)), overwritten):
        test_generator = SwappedMultiOutputHDF5Generator(data_2022, 
        batch_size=BATCH_SIZE,
        fields=config['fields'],
        num_target_classes=config['num_outputs'],
        lower_boundary=config['lower_boundary'],
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['test_normalization']}.csv",
        seed = seed_value,
        augment=config['test_augment'],
        shuffle=config['test_shuffle'],
        swap_index=i
        )

        predict_test_multi(test_generator, model, PATH_OUTPUTS, weights, config['lead_time'], name, seed_value)




if __name__ == "__main__":
    main()