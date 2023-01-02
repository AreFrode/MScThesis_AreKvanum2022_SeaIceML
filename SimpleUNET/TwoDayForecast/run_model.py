import sys
# sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET")
sys.path.append("/mnt/SimpleUNET")

import glob
import os
import csv
import time
import h5py
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET, create_MultiOutputUNET
from dataset import HDF5Generator, MultiOutputHDF5Generator
from focalLoss import categorical_focal_loss
from customCallbacks import MemoryPrintingCallback, IIEECallback
from helper_functions import read_ice_edge_from_csv

def main():
    current_time = datetime.now().strftime("%d%m%H%M")
    print(f"Time at start of script {current_time}")

    keras.mixed_precision.set_global_policy('mixed_float16')

    SEED_VALUE = 0
    # PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
    # PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    # log_dir = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/logs/fit/{datetime.now().strftime('%d%m%H%M')}"

    # Comment above and uncomment below if running tensorflow2.11-singularity container
    PATH_OUTPUT = "/mnt/SimpleUNET/TwoDayForecast/outputs/"
    PATH_DATA = "/mnt/PrepareDataset/Data/two_day_forecast/"
    # PATH_CLIMATOLOGICAL_ICEEDGE = "/mnt/verification_metrics/Data/climatological_ice_edge.csv"
    log_dir = f"/mnt/SimpleUNET/TwoDayForecast/logs/fit/{datetime.now().strftime('%d%m%H%M')}"

    # THIS SHOULD BE WHERE I NEED TO EDIT FOR EXPERIMENTS
    config = {
        'BATCH_SIZE': 4,
        'constant_fields': ['sic', 'sic_trend', 'lsmask'],
        'dated_fields': ['t2m', 'xwind', 'ywind'],
        'train_augment': False,
        'train_normalization': 'normalization_constants_train',
        'train_shuffle': True,
        'val_augment': False,
        'val_normalization': 'normalization_constants_validation',
        'val_shuffle': False,
        'learning_rate': 0.001,
        'epochs': 40,
        'pooling_factor': 4,
        'channels': [64, 128, 256, 512],
        'height': 1792,
        'width': 1792,
        'lower_boundary': 578,
        'rightmost_boundary': 1792,
        'model_name': f'weights_{current_time}',
        'GroupNorm': True,
        'AveragePool': False,
        'LeakyReLU': False,
        'ResidualUNET': True
    }

    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    
    data_2019 = np.array(sorted(glob.glob(f"{PATH_DATA}2019/**/*.hdf5")))
    data_2020 = np.array(sorted(glob.glob(f"{PATH_DATA}2020/**/*.hdf5")))
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5")))

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    if not os.path.exists(f"{PATH_OUTPUT}models"):
        os.makedirs(f"{PATH_OUTPUT}models")

    if not os.path.exists(f"{PATH_OUTPUT}configs"):
        os.makedirs(f"{PATH_OUTPUT}configs")

    train_generator = MultiOutputHDF5Generator(
        np.concatenate((data_2019, data_2020)), 
        batch_size=config['BATCH_SIZE'], 
        constant_fields=config['constant_fields'], 
        dated_fields=config['dated_fields'], 
        lower_boundary=config['lower_boundary'], 
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['train_normalization']}.csv",
        shuffle=config['train_shuffle'],
        augment=config['train_augment']
    )

    val_generator = MultiOutputHDF5Generator(
        data_2021, 
        batch_size=config['BATCH_SIZE'], 
        constant_fields=config['constant_fields'], 
        dated_fields=config['dated_fields'], 
        lower_boundary=config['lower_boundary'], 
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['val_normalization']}.csv",
        shuffle=config['val_shuffle'],
        augment=config['val_augment']
    )
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config['learning_rate'],
        decay_steps = 500,
        decay_rate = .96,
        staircase=True
    )

    # model = create_UNET(input_shape = (1920, 1840, 9), channels = [64, 128, 256, 512])
    model = create_MultiOutputUNET(
        input_shape = (config['height'], config['width'], len(config['constant_fields']) + 2*len(config['dated_fields'])), 
        channels = config['channels'],
        pooling_factor= config['pooling_factor'],
        average_pool=config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)

    loss_function = keras.losses.CategoricalCrossentropy()
    loss_function_multi = keras.losses.BinaryCrossentropy(from_logits = True)
    focal_loss_function = categorical_focal_loss(
        alpha=np.expand_dims(0.25*np.ones(7),0), 
        gamma=2
    )
        
    # model.compile(optimizer = optimizer, loss = [focal_loss_function], metrics=['accuracy'])
    model.compile(
        optimizer = optimizer, 
        loss = loss_function_multi, 
        metrics=['accuracy']
    )

    model.summary()

    checkpoint_filepath = f"{PATH_OUTPUT}models/{config['model_name']}"

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = True,
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )

    if not os.path.exists(f"{PATH_OUTPUT}histories/"):
        os.makedirs(f"{PATH_OUTPUT}histories/")

    csvlogger_callback = keras.callbacks.CSVLogger(
        filename = f"{PATH_OUTPUT}histories/{config['model_name']}.log"
    )

    memory_print_callback = MemoryPrintingCallback()

    ########### SETUP IIEECallback #################
    """
    # The following code is only necesarry if computing IIEE each epoch
    with h5py.File(data_2019[0], 'r') as infile:
        lsmask = infile['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]

    iiee_val_generator = MultiOutputHDF5Generator(
        data_2021, 
        batch_size=1, 
        constant_fields=config['constant_fields'], 
        dated_fields=config['dated_fields'], 
        lower_boundary=config['lower_boundary'], 
        rightmost_boundary=config['rightmost_boundary'],
        normalization_file=f"{PATH_DATA}{config['val_normalization']}.csv",
        shuffle=config['val_shuffle'],
        augment=config['val_augment']
    )

    climatoloigcal_ice_edge = read_ice_edge_from_csv(PATH_CLIMATOLOGICAL_ICEEDGE)

    iiee_callback = IIEECallback(
        validation_data = iiee_val_generator,
        lsmask = lsmask,
        batch_size = 1,
        ice_edge = climatoloigcal_ice_edge
    )
    """
    ######## END SETUP IIEECallback #################

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    t0 = time.time()
    model.fit(
        train_generator,
        epochs = config['epochs'],
        batch_size = config['BATCH_SIZE'],
        callbacks=[
            model_checkpoint_callback, 
            tensorboard_callback,
            memory_print_callback,
            # iiee_callback,   # REMOVE IF NOT COMPUTING IIEE every epoch
            csvlogger_callback
        ],
        validation_data = val_generator
    )
    config['fit_runtime'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))

    with open(f"{PATH_OUTPUT}configs/{config['model_name']}.csv", 'w') as f:
        w = csv.DictWriter(f, config.keys())
        w.writeheader()
        w.writerow(config)
    

if __name__ == "__main__":
    main()