import sys
# sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/mnt/SimpleUNET")
sys.path.append("/mnt/SimpleUNET/RunModel")

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
from predict_validation import numpy_where_wrapper
from netCDF4 import Dataset

from matplotlib import pyplot as plt


def min_max(X, min, max):
    return (X - min) / (max - min)


def main():    
    SEED_VALUE = 0
    # PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    # PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    assert len(sys.argv) > 1, "Remember to provide weights"
    weights = sys.argv[1]
    
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

    test_generator = MultiOutputHDF5Generator(data_2022, 
        batch_size=BATCH_SIZE,
        fields=config['fields'],
        num_target_classes=config['num_outputs'],
        lower_boundary=config['lower_boundary'],
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['test_normalization']}.csv",
        augment=config['test_augment'],
        shuffle=config['test_shuffle']
    )


    synthetic_forcings = np.empty((6, *test_generator[0][0][0].shape[:2]))

    increasing_t2m_west_east = np.tile(np.linspace(test_generator.mins['t2m'], test_generator.maxs['t2m'], num = config['width']), (config['height'], 1))

    increasing_t2m_east_west = np.tile(np.linspace(test_generator.maxs['t2m'], test_generator.mins['t2m'], num = config['width']), (config['height'], 1))

    increasing_t2m_south_north = increasing_t2m_west_east.T
    increasing_t2m_north_south = increasing_t2m_east_west.T

    uniform_t2m_min = test_generator.mins['t2m'] * np.ones_like(increasing_t2m_east_west)
    uniform_t2m_max = test_generator.maxs['t2m'] * np.ones_like(increasing_t2m_east_west)

    synthetic_forcings[0] = increasing_t2m_west_east
    synthetic_forcings[1] = increasing_t2m_east_west
    synthetic_forcings[2] = increasing_t2m_south_north
    synthetic_forcings[3] = increasing_t2m_north_south
    synthetic_forcings[4] = uniform_t2m_min
    synthetic_forcings[5] = uniform_t2m_max

    names = ['west-east', 'east-west', 'south-north', 'north-south', 'uniform-min', 'uniform-max']


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

    dates = np.array([datetime.strptime(test_generator.get_dates(i)[0][-13:-5], '%Y%m%d') for i in range(len(test_generator))])
    mar = [d for d in dates if d.month == 3][0]
    jun = [d for d in dates if d.month == 6][0]
    sep = [d for d in dates if d.month == 9][0]
    des = [d for d in dates if d.month == 12][0]

    inspect_dates = [np.argwhere(dates == mar)[0][0], np.argwhere(dates == jun)[0][0], np.argwhere(dates == sep)[0][0], np.argwhere(dates == des)[0][0]]

    """
    for date in inspect_dates:
        for field, name in zip(synthetic_forcings, names):
            X, y = test_generator[date]
            
            normalized_field = min_max(field, test_generator.mins['t2m'], test_generator.maxs['t2m'])
            X[...,3] = normalized_field

            y_pred = model.predict(X)
            y_pred = tf.concat(y_pred, axis=-1)
            
            yyyymmdd = test_generator.get_dates(date)[0][-13:-5]
            yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
            yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = config['lead_time'])).strftime('%Y%m%d')
            
            # out = tf.math.reduce_sum(tf.round(tf.nn.sigmoid(y_pred[0])), -1)
            out = tf.round(tf.nn.sigmoid(y_pred[0]))

            concentration_and_changes = np.apply_along_axis(numpy_where_wrapper, -1, out)

            out = concentration_and_changes[...,0]

            x_vals, y_vals = test_generator.get_xy(date)

            hdf_path = f"/mnt/ForecastValidation/SyntheticForcing/Data/{weights}/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
            if not os.path.exists(hdf_path):
                os.makedirs(hdf_path)

            with Dataset(f"{hdf_path}SIC_UNET_t2m_{date}_{name}_v{yyyymmdd_valid}_b{yyyymmdd}T15Z.nc", "w") as output_file:
                output_file.createDimension('x', len(x_vals))
                output_file.createDimension('y', len(y_vals))

                out_x = output_file.createVariable('xc', 'd', ('x'))
                out_x[:] = x_vals

                out_y = output_file.createVariable('yc', 'd', ('y'))
                out_y[:] = y_vals

                out_ypred = output_file.createVariable('y_pred', 'd', ('y', 'x'))
                out_ypred[:] = np.expand_dims(out, 0)

                out_t2mfield = output_file.createVariable('t2m_mod', 'd', ('y', 'x'))
                out_t2mfield[:] = X[...,3]
    """

    synthetic_winds = np.empty((7, 2, *test_generator[0][0][0].shape[:2]))

    no_wind = np.zeros_like(synthetic_forcings[0,0])

    max_xwind = test_generator.maxs['xwind'] * np.ones_like(no_wind)
    min_xwind = test_generator.mins['xwind'] * np.ones_like(no_wind)

    max_ywind = test_generator.maxs['ywind'] * np.ones_like(no_wind)
    min_ywind = test_generator.mins['ywind'] * np.ones_like(no_wind)

    synthetic_winds[0,0] = no_wind
    synthetic_winds[0,1] = no_wind

    synthetic_winds[1,0] = max_xwind
    synthetic_winds[1,1] = no_wind

    synthetic_winds[2,0] = min_xwind
    synthetic_winds[2,1] = no_wind

    synthetic_winds[3,0] = no_wind
    synthetic_winds[3,1] = max_ywind

    synthetic_winds[4,0] = no_wind
    synthetic_winds[4,1] = min_ywind

    synthetic_winds[5,0] = max_xwind
    synthetic_winds[5,1] = max_ywind

    synthetic_winds[6,0] = min_xwind
    synthetic_winds[6,1] = min_ywind

    names_wind = ['no-wind', 'only_xwind_pos', 'only_xwind_neg', 'only_ywind_pos', 'only_ywind_neg', 'both_winds_pos', 'both_winds_neg']

    for date in inspect_dates:
        for field, name in zip(synthetic_winds, names_wind):
            X, y = test_generator[date]
            
            normalized_xfield = min_max(field[0], test_generator.mins['xwind'], test_generator.maxs['xwind'])
            X[...,4] = normalized_xfield

            normalized_yfield = min_max(field[1], test_generator.mins['ywind'], test_generator.maxs['ywind'])
            X[...,5] = normalized_yfield

            y_pred = model.predict(X)
            y_pred = tf.concat(y_pred, axis=-1)
            
            yyyymmdd = test_generator.get_dates(date)[0][-13:-5]
            yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
            yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = config['lead_time'])).strftime('%Y%m%d')
            
            # out = tf.math.reduce_sum(tf.round(tf.nn.sigmoid(y_pred[0])), -1)
            out = tf.round(tf.nn.sigmoid(y_pred[0]))

            concentration_and_changes = np.apply_along_axis(numpy_where_wrapper, -1, out)

            out = concentration_and_changes[...,0]

            x_vals, y_vals = test_generator.get_xy(date)

            hdf_path = f"/mnt/ForecastValidation/SyntheticForcing/Data/{weights}/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
            if not os.path.exists(hdf_path):
                os.makedirs(hdf_path)

            with Dataset(f"{hdf_path}SIC_UNET_winds_{date}_{name}_v{yyyymmdd_valid}_b{yyyymmdd}T15Z.nc", "w") as output_file:
                output_file.createDimension('x', len(x_vals))
                output_file.createDimension('y', len(y_vals))

                out_x = output_file.createVariable('xc', 'd', ('x'))
                out_x[:] = x_vals

                out_y = output_file.createVariable('yc', 'd', ('y'))
                out_y[:] = y_vals

                out_ypred = output_file.createVariable('y_pred', 'd', ('y', 'x'))
                out_ypred[:] = np.expand_dims(out, 0)

                out_xwindfield = output_file.createVariable('xwind_mod', 'd', ('y', 'x'))
                out_xwindfield[:] = X[...,4]

                out_ywindfield = output_file.createVariable('ywind_mod', 'd', ('y', 'x'))
                out_ywindfield[:] = X[...,5]


if __name__ == "__main__":
    main()