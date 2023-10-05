import sys
sys.path.append("/mnt/SimpleUNET")
sys.path.append("/mnt/SimpleUNET/RunModel")
sys.path.append("/mnt/CreateFigures")

import glob
import cv2
import WMOcolors
import cmocean
import h5py

import tensorflow as tf
import numpy as np

from netCDF4 import Dataset
from unet import create_MultiOutputUNET_seggradcam, Encoder_Model, create_Decoder_Model
from helper_functions import read_config_from_csv
from dataset import MultiOutputHDF5Generator
from predict_validation import numpy_where_wrapper



from matplotlib import pyplot as plt, colors as mcolors



weights = "weights_21021550"
weights_not2m = "weights_09031802"
weights_r = "weights_01031920"



config = read_config_from_csv(f"/mnt/SimpleUNET/RunModel/outputs/configs/{weights}.csv")
config_not2m = read_config_from_csv(f"/mnt/SimpleUNET/RunModel/outputs/configs/{weights_not2m}.csv")
config_r = read_config_from_csv(f"/mnt/SimpleUNET/RunModel/outputs/configs/{weights_r}.csv")

path = f"/mnt/PrepareDataset/Data/lead_time_{config['lead_time']}/2022/"
h5file = sorted(glob.glob(f"{path}/01/*.hdf5"))[0]

with h5py.File(h5file, 'r') as f:
    lat = f['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
    lon = f['lon'][config['lower_boundary']:, :config['rightmost_boundary']]
    lsmask = f['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]



PATH_DATA = f"/mnt/PrepareDataset/Data/lead_time_{config['lead_time']}/"

PATH_DATA_R = f"/mnt/PrepareDataset/Data/reduced_classes/lead_time_{config_r['lead_time']}/"

data_2022 = np.array(sorted(glob.glob(f"{PATH_DATA}2022/**/*.hdf5")))
data_2022_reduced = np.array(sorted(glob.glob(f"{PATH_DATA_R}2022/**/*.hdf5")))


samples = [0, 24, 59, 98, 136]

data_generator = MultiOutputHDF5Generator(data_2022, 
                                          batch_size = 1,
                                          fields = config['fields'],
                                          num_target_classes = config['num_outputs'],
                                          lower_boundary = config['lower_boundary'],
                                          rightmost_boundary = config['rightmost_boundary'],
                                          normalization_file=f"{PATH_DATA}{config['test_normalization']}.csv",
                                          augment = False,
                                          shuffle = False)


data_generator_not2m = MultiOutputHDF5Generator(data_2022, 
                                          batch_size = 1,
                                          fields = config_not2m['fields'],
                                          num_target_classes = config_not2m['num_outputs'],
                                          lower_boundary = config_not2m['lower_boundary'],
                                          rightmost_boundary = config_not2m['rightmost_boundary'],
                                          normalization_file=f"{PATH_DATA}{config_not2m['test_normalization']}.csv",
                                          augment = False,
                                          shuffle = False)

data_generator_reduced = MultiOutputHDF5Generator(data_2022_reduced, 
                                          batch_size = 1,
                                          fields = config_r['fields'],
                                          num_target_classes = config_r['num_outputs'],
                                          lower_boundary = config_r['lower_boundary'],
                                          rightmost_boundary = config_r['rightmost_boundary'],
                                          normalization_file=f"{PATH_DATA_R}{config_r['test_normalization']}.csv",
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

model_not2m = create_MultiOutputUNET_seggradcam(
        input_shape = (config_not2m['height'], config_not2m['width'], len(config_not2m['fields'])), 
        channels = config_not2m['channels'],
        pooling_factor = config_not2m['pooling_factor'],
        num_outputs = config_not2m['num_outputs'],
        # num_outputs = 1,
        average_pool = config_not2m['AveragePool'],
        leaky_relu = config_not2m['LeakyReLU']
    )

model_r = create_MultiOutputUNET_seggradcam(
        input_shape = (config_r['height'], config_r['width'], len(config_r['fields'])), 
        channels = config_r['channels'],
        pooling_factor = config_r['pooling_factor'],
        num_outputs = config_r['num_outputs'],
        # num_outputs = 1,
        average_pool = config_r['AveragePool'],
        leaky_relu = config_r['LeakyReLU']
    )



model_singleoutput = create_MultiOutputUNET_seggradcam(
        input_shape = (config['height'], config['width'], len(config['fields'])), 
        channels = config['channels'],
        pooling_factor = config['pooling_factor'],
        num_outputs = 1,
        average_pool = config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

model_not2m_singleoutput = create_MultiOutputUNET_seggradcam(
        input_shape = (config_not2m['height'], config_not2m['width'], len(config_not2m['fields'])), 
        channels = config_not2m['channels'],
        pooling_factor = config_not2m['pooling_factor'],
        num_outputs = 1,
        average_pool = config_not2m['AveragePool'],
        leaky_relu = config_not2m['LeakyReLU']
    )

model_r_singleoutput = create_MultiOutputUNET_seggradcam(
        input_shape = (config_r['height'], config_r['width'], len(config_r['fields'])), 
        channels = config_r['channels'],
        pooling_factor = config_r['pooling_factor'],
        num_outputs = 1,
        average_pool = config_r['AveragePool'],
        leaky_relu = config_r['LeakyReLU']
    )

model.load_weights(f"/mnt/SimpleUNET/RunModel/outputs/models/{weights}").expect_partial()
model_not2m.load_weights(f"/mnt/SimpleUNET/RunModel/outputs/models/{weights_not2m}").expect_partial()
model_r.load_weights(f"/mnt/SimpleUNET/RunModel/outputs/models/{weights_r}").expect_partial()

# Transfer weights from multioutput model to singleoutput model
model_singleoutput.layers[0].set_weights(model.layers[0].get_weights())
model_singleoutput.layers[1].layers[0].set_weights(model.layers[1].layers[0].get_weights())
model_singleoutput.layers[1].layers[1].set_weights(model.layers[1].layers[1].get_weights())
model_singleoutput.layers[1].layers[2].set_weights(model.layers[1].layers[2].get_weights())

model_not2m_singleoutput.layers[0].set_weights(model_not2m.layers[0].get_weights())
model_not2m_singleoutput.layers[1].layers[0].set_weights(model_not2m.layers[1].layers[0].get_weights())
model_not2m_singleoutput.layers[1].layers[1].set_weights(model_not2m.layers[1].layers[1].get_weights())
model_not2m_singleoutput.layers[1].layers[2].set_weights(model_not2m.layers[1].layers[2].get_weights())

model_r_singleoutput.layers[0].set_weights(model_r.layers[0].get_weights())
model_r_singleoutput.layers[1].layers[0].set_weights(model_r.layers[1].layers[0].get_weights())
model_r_singleoutput.layers[1].layers[1].set_weights(model_r.layers[1].layers[1].get_weights())
model_r_singleoutput.layers[1].layers[2].set_weights(model_r.layers[1].layers[2].get_weights())




# Extract sample from dataloader
sample = samples[0]

# Figure I and II
cams = []
cams_r = []

for i in range(5, 9):
    model_singleoutput.layers[1].layers[3].set_weights(model.layers[1].layers[i].get_weights())

    X, y = data_generator[sample]
    print(X.shape)

    y = np.array(y)
    print(y.shape)

    # Predict logits for single class >10% contour
    single_y_pred, fmaps = model_singleoutput.predict(X)
    single_y_pred = single_y_pred[0]

    # Create roi
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

    cams.append(cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR))


    model_r_singleoutput.layers[1].layers[3].set_weights(model_r.layers[1].layers[i-1].get_weights())

    X, y = data_generator_reduced[sample]
    print(X.shape)

    y = np.array(y)
    print(y.shape)

    # Predict logits for single class >10% contour
    single_y_pred, fmaps = model_r_singleoutput.predict(X)
    single_y_pred = single_y_pred[0]

    # Create roi
    single_out = tf.round(tf.nn.sigmoid(single_y_pred))
    roi = single_out

    # Setup decoder only network
    decoder = create_Decoder_Model(input_shape = (112, 112, 256),
                               feature_maps = fmaps,
                               channels = config_r['channels'],
                               pooling_factor = config_r['pooling_factor'],
                               num_outputs = 1,
                               average_pool = config_r['AveragePool'],
                               leaky_relu = config_r['LeakyReLU']
    )



    decoder.layers[1].layers[0].set_weights(model_r_singleoutput.layers[1].layers[1].get_weights())
    decoder.layers[1].layers[1].set_weights(model_r_singleoutput.layers[1].layers[3].get_weights())

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

    cams_r.append(cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR))

# Figure III and IV
cams_3 = []
cams_4 = []

model_singleoutput.layers[1].layers[3].set_weights(model.layers[1].layers[5].get_weights())
model_not2m_singleoutput.layers[1].layers[3].set_weights(model_not2m.layers[1].layers[5].get_weights())

for sample in samples[1:]:
    X, y = data_generator[sample]
    print(X.shape)

    y = np.array(y)
    print(y.shape)

    # Predict logits for single class >10% contour
    single_y_pred, fmaps = model_singleoutput.predict(X)
    single_y_pred = single_y_pred[0]

    # Create roi
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

    cams_3.append(cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR))

    X, y = data_generator_not2m[sample]
    print(X.shape)

    y = np.array(y)
    print(y.shape)

    # Predict logits for single class >10% contour
    single_y_pred, fmaps = model_not2m_singleoutput.predict(X)
    single_y_pred = single_y_pred[0]

    # Create roi
    single_out = tf.round(tf.nn.sigmoid(single_y_pred))
    roi = single_out

    # Setup decoder only network
    decoder = create_Decoder_Model(input_shape = (112, 112, 256),
                               feature_maps = fmaps,
                               channels = config_not2m['channels'],
                               pooling_factor = config_not2m['pooling_factor'],
                               num_outputs = 1,
                               average_pool = config_not2m['AveragePool'],
                               leaky_relu = config_not2m['LeakyReLU']
    )



    decoder.layers[1].layers[0].set_weights(model_not2m_singleoutput.layers[1].layers[1].get_weights())
    decoder.layers[1].layers[1].set_weights(model_not2m_singleoutput.layers[1].layers[3].get_weights())

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

    cams_4.append(cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR))



with Dataset('figure_1.nc', 'w') as out1:
    out1.createDimension('x', config['width'])
    out1.createDimension('y', config['height'])
    out1.createDimension('t', 4)

    lat1 = out1.createVariable('lat', 'd', ('y', 'x'))
    lat1[:] = lat
    
    lon1 = out1.createVariable('lon', 'd', ('y', 'x'))
    lon1[:] = lon

    cam1 = out1.createVariable('cam', 'd', ('t', 'y', 'x'))
    cam1[:] = cams

with Dataset('figure_2.nc', 'w') as out2:
    out2.createDimension('x', config['width'])
    out2.createDimension('y', config['height'])
    out2.createDimension('t', 4)

    lat2 = out2.createVariable('lat', 'd', ('y', 'x'))
    lat2[:] = lat
    
    lon2 = out2.createVariable('lon', 'd', ('y', 'x'))
    lon2[:] = lon

    cam2 = out2.createVariable('cam', 'd', ('t', 'y', 'x'))
    cam2[:] = cams_r

with Dataset('figure_3.nc', 'w') as out3:
    out3.createDimension('x', config['width'])
    out3.createDimension('y', config['height'])
    out3.createDimension('t', 4)

    lat3 = out3.createVariable('lat', 'd', ('y', 'x'))
    lat3[:] = lat
    
    lon3 = out3.createVariable('lon', 'd', ('y', 'x'))
    lon3[:] = lon

    cam3 = out3.createVariable('cam', 'd', ('t', 'y', 'x'))
    cam3[:] = cams_3

with Dataset('figure_4.nc', 'w') as out4:
    out4.createDimension('x', config['width'])
    out4.createDimension('y', config['height'])
    out4.createDimension('t', 4)

    lat4 = out4.createVariable('lat', 'd', ('y', 'x'))
    lat4[:] = lat
    
    lon4 = out4.createVariable('lon', 'd', ('y', 'x'))
    lon4[:] = lon

    cam4 = out4.createVariable('cam', 'd', ('t', 'y', 'x'))
    cam4[:] = cams_4