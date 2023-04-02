import sys
sys.path.append("/mnt/verification_metrics")
sys.path.append("/mnt/PrepareDataset")
sys.path.append("/mnt/ForecastValidation/Forecasts")
sys.path.append("/mnt/PhysicalModels")
sys.path.append("/mnt/SimpleUNET")
sys.path.append("/mnt/SimpleUNET/RunModel")
sys.path.append("/mnt/CreateFigures")

import glob
import os
import cv2

import numpy as np
import tensorflow as tf
import pandas as pd

from netCDF4 import Dataset
from tqdm import tqdm
from verification_metrics import IIEE_alt, find_ice_edge, find_ice_edge_from_fraction, ice_edge_length, calculate_distance

from scipy.interpolate import NearestNDInterpolator
from datetime import datetime
from dateutil.relativedelta import relativedelta

from matplotlib import pyplot as plt

from unet import create_MultiOutputUNET_seggradcam, Encoder_Model, create_Decoder_Model
from helper_functions import read_config_from_csv
from dataset import MultiOutputHDF5Generator
from predict_validation import numpy_where_wrapper


def find_ice_edges(config, data_2022, data_generator, model_singleoutput):
    PATH_AROME = "/mnt/AROME_ARCTIC_regrid/Data/"
    OUT_PATH = "/mnt/Saliency/Contour/"

    baltic_mask = np.zeros((2370, 1845))
    baltic_mask[:1200, 1500:] = 1

    rows = []

    for i in range(len(data_generator)):
        path_bulletin = data_2022[i]
        yyyymmdd_bulletin = data_generator.get_dates(i)[0][-13:-5]

        # yyyymmdd_bulletin_dt = datetime.strptime(yyyymmdd_bulletin, '%Y%m%d')
        # yyyymmdd_valid_dt = yyyymmdd_bulletin_dt + relativedelta(days=+2)
        # yyyymmdd_valid = datetime.strftime(yyyymmdd_valid_dt, '%Y%m%d')

        try:
            arome = glob.glob(f"{PATH_AROME}{yyyymmdd_bulletin[:4]}/{yyyymmdd_bulletin[4:6]}/AROME_1kmgrid_{yyyymmdd_bulletin}T18Z.nc")[0]
            # ml = glob.glob(f"{PATH_DATA}{yyyymmdd_bulletin[:4]}/{yyyymmdd_bulletin[4:6]}/PreparedSample_v{yyyymmdd_valid}_b{yyyymmdd_bulletin}.hdf5")[0]

        except IndexError:
            continue

        with Dataset(arome, 'r') as nc_bul:
            # sic_arome = onehot_encode_sic(np.nan_to_num(nc_bul.variables['sic'][:], nan=7.))
            sic_arome = nc_bul.variables['sic'][:]
            lat_arome = nc_bul.variables['lat'][578:, :1792]
            lon_arome = nc_bul.variables['lon'][578:, :1792]
            x_arome = nc_bul.variables['x'][:1792]
            y_arome = nc_bul.variables['y'][578:]

        # sic_bulletin = np.where(mask == 1, sic_bulletin, 0)
        mask = np.where(~np.logical_or((baltic_mask == 1), (sic_arome == -10)))
        mask_T = np.transpose(mask)
        sic_arome_interpolator = NearestNDInterpolator(mask_T, sic_arome[mask])
        sic_arome = sic_arome_interpolator(*np.indices(sic_arome.shape))

        X, y = data_generator[i]
        y = np.array(y)

        # with Dataset(ml, 'r') as nc:
            # sic_target = nc.variables['sic'][578:, :1792]
            # lsmask = nc.variables['lsmask'][578:, :1792]
            # lat = nc.variables['lat'][578:, :1792]
            # lon = nc.variables['lon'][578:, :1792]

        sic_arome = sic_arome[578:, :1792]

        sic_icechart = (6*X[0, ..., 0]).astype(int)
        lsmask = X[0, ..., 2].astype(int)

        single_y_pred, fmaps = model_singleoutput.predict(X)
        single_y_pred = single_y_pred[0]

        single_out = tf.round(tf.nn.sigmoid(single_y_pred))
        roi = single_out 

        # Setup decoder only network
        decoder = create_Decoder_Model(input_shape = (112, 112, 256),
                               feature_maps = fmaps,
                               channels = config['channels'],
                               pooling_factor = config['pooling_factor'],
                               num_outputs = 1,
                               average_pool = config['AveragePool'],
                               leaky_relu = config['LeakyReLU']
        )
        decoder.layers[1].layers[0].set_weights(model_singleoutput.layers[1].layers[1].get_weights())
        decoder.layers[1].layers[1].set_weights(model_singleoutput.layers[1].layers[3].get_weights())

        # Compute gradient w.r.t. bottleneck
        bottleneck = tf.Variable(fmaps[0])

        with tf.GradientTape() as tape:
            tape.watch(bottleneck)
            decoder_pred = decoder(bottleneck, training = False)[0]

            y_c = tf.where(roi == 1, decoder_pred, 0)
            y_c = tf.math.reduce_sum(y_c)

        grads = tape.gradient(y_c, bottleneck)[0]

        # Compute seg-grad-cam
        alpha = np.mean(grads, axis=(0,1))

        unactivated_cam = np.dot(bottleneck, alpha)

        cam = np.maximum(unactivated_cam, 0)

        # Normalize
        cam = cam / cam.max()

        cam = np.moveaxis(cam, 0, -1)

        resized_cam = cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR)

        min_value = np.unique(resized_cam)[1]

        # plt.figure()
        # plt.pcolormesh(np.ma.masked_where(resized_cam == 0, resized_cam), cmap = 'jet')
        # plt.savefig('test5')

        # lsmask = X[]

        # print('finding ice edges')

        # arome_ice_edge = find_ice_edge(sic_arome, lsmask, threshold = 2)
        arome_ice_edge = find_ice_edge_from_fraction(sic_arome, lsmask, threshold = 0.1)
        # arome_ice_edge_length = ice_edge_length(arome_ice_edge, s = 1)


        icechart_ice_edge = find_ice_edge(sic_icechart, lsmask, threshold = 2)
        # icechart_ice_edge_length = ice_edge_length(icechart_ice_edge, s = 1)

        saliency_ice_edge = find_ice_edge_from_fraction(resized_cam, np.zeros_like(lsmask), threshold = min_value)

        # print('calculating distance arome saliency')

        arome_to_sal = calculate_distance(arome_ice_edge, saliency_ice_edge, x_arome, y_arome)
        sal_to_arome = calculate_distance(saliency_ice_edge, arome_ice_edge, x_arome, y_arome)

        arome_displacement = 0.5*(np.mean(arome_to_sal['distance']) + np.mean(sal_to_arome['distance']))

        icechart_to_sal = calculate_distance(icechart_ice_edge, saliency_ice_edge, x_arome, y_arome)
        sal_to_icechart = calculate_distance(saliency_ice_edge, icechart_ice_edge, x_arome, y_arome)

        icechart_displacement = 0.5*(np.mean(icechart_to_sal['distance']) + np.mean(sal_to_icechart['distance']))

        # plt.figure()
        # plt.scatter(lon, lat, 0.05*arome_ice_edge, label = 'arome')
        # plt.scatter(lon, lat, 0.05*icechart_ice_edge, label = 'icechart')
        # plt.scatter(lon, lat, 0.05*saliency_ice_edge, label = 'saliency')
        # plt.pcolormesh(arome_ice_edge, label = 'arome')
        # plt.pcolormesh(icechart_ice_edge, label = 'icechart')
        # plt.legend()
        # plt.savefig('test6')
        # exit()

        

        rows.append([yyyymmdd_bulletin, arome_displacement, icechart_displacement])
        

    df = pd.DataFrame(columns = ['date', 'arome_displacement', 'icechart_displacement'], data = rows)
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")

    df.set_index('date', inplace = True)

    df.to_csv(f"{OUT_PATH}displacement_to_saliency.csv")

    

def main():
    config = read_config_from_csv(f"/mnt/SimpleUNET/RunModel/outputs/configs/weights_21021550.csv")

    PATH_DATA = f"/mnt/PrepareDataset/Data/lead_time_{config['lead_time']}/"

    data_2022 = np.array(sorted(glob.glob(f"{PATH_DATA}2022/**/*.hdf5")))


    data_generator = MultiOutputHDF5Generator(data_2022, 
                                              batch_size = 1,
                                              fields = config['fields'],
                                              num_target_classes = config['num_outputs'],
                                              lower_boundary = config['lower_boundary'],
                                              rightmost_boundary = config['rightmost_boundary'],
                                              normalization_file=f"{PATH_DATA}{config['test_normalization']}.csv",
                                              augment = False,
                                              shuffle = False)
      



    model = create_MultiOutputUNET_seggradcam(
            input_shape = (config['height'], config['width'], len(config['fields'])), 
            channels = config['channels'],
            pooling_factor = config['pooling_factor'],
            num_outputs = config['num_outputs'],
            # num_outputs = 1,
            average_pool = config['AveragePool'],
            leaky_relu = config['LeakyReLU']
        )
    

    model_singleoutput = create_MultiOutputUNET_seggradcam(
        input_shape = (config['height'], config['width'], len(config['fields'])), 
        channels = config['channels'],
        pooling_factor = config['pooling_factor'],
        num_outputs = 1,
        average_pool = config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

    load_status = model.load_weights(f"/mnt/SimpleUNET/RunModel/outputs/models/weights_21021550").expect_partial()

    # Transfer weights from multioutput model to singleoutput model
    model_singleoutput.layers[0].set_weights(model.layers[0].get_weights())
    model_singleoutput.layers[1].layers[0].set_weights(model.layers[1].layers[0].get_weights())
    model_singleoutput.layers[1].layers[1].set_weights(model.layers[1].layers[1].get_weights())
    model_singleoutput.layers[1].layers[2].set_weights(model.layers[1].layers[2].get_weights())
    model_singleoutput.layers[1].layers[3].set_weights(model.layers[1].layers[5].get_weights())


    find_ice_edges(config, data_2022, data_generator, model_singleoutput)



if __name__ == "__main__":
    main()