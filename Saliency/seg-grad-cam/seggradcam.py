import sys
sys.path.append("/mnt/SimpleUNET")
sys.path.append("/mnt/SimpleUNET/RunModel")
sys.path.append("/mnt/CreateFigures")

import glob
import cv2
import WMOcolors
import cmocean

import tensorflow as tf
import numpy as np

from unet import create_MultiOutputUNET_seggradcam, Encoder_Model, create_Decoder_Model
from helper_functions import read_config_from_csv
from dataset import MultiOutputHDF5Generator
from predict_validation import numpy_where_wrapper



from matplotlib import pyplot as plt, colors as mcolors

# weights = "weights_21021550"
# weights = "weights_09031802"
weights = "weights_10021139"
config = read_config_from_csv(f"/mnt/SimpleUNET/RunModel/outputs/configs/{weights}.csv")



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
exit()

# print(model.summary())
# print(model.get_layer('unet').summary())
# print(model.get_layer('unet').get_layer('encoder').layers)

model_singleoutput = create_MultiOutputUNET_seggradcam(
        input_shape = (config['height'], config['width'], len(config['fields'])), 
        channels = config['channels'],
        pooling_factor = config['pooling_factor'],
        num_outputs = 1,
        average_pool = config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

load_status = model.load_weights(f"/mnt/SimpleUNET/RunModel/outputs/models/{weights}").expect_partial()

# Transfer weights from multioutput model to singleoutput model
model_singleoutput.layers[0].set_weights(model.layers[0].get_weights())
model_singleoutput.layers[1].layers[0].set_weights(model.layers[1].layers[0].get_weights())
model_singleoutput.layers[1].layers[1].set_weights(model.layers[1].layers[1].get_weights())
model_singleoutput.layers[1].layers[2].set_weights(model.layers[1].layers[2].get_weights())
model_singleoutput.layers[1].layers[3].set_weights(model.layers[1].layers[5].get_weights()) # 10%
# model_singleoutput.layers[1].layers[3].set_weights(model.layers[1].layers[8].get_weights()) # 90%

# Create new model to find difference
model_singleoutput2 = create_MultiOutputUNET_seggradcam(
        input_shape = (config['height'], config['width'], len(config['fields'])), 
        channels = config['channels'],
        pooling_factor = config['pooling_factor'],
        num_outputs = 1,
        average_pool = config['AveragePool'],
        leaky_relu = config['LeakyReLU']
    )

model_singleoutput2.layers[0].set_weights(model.layers[0].get_weights())
model_singleoutput2.layers[1].layers[0].set_weights(model.layers[1].layers[0].get_weights())
model_singleoutput2.layers[1].layers[1].set_weights(model.layers[1].layers[1].get_weights())
model_singleoutput2.layers[1].layers[2].set_weights(model.layers[1].layers[2].get_weights())
# model_singleoutput2.layers[1].layers[3].set_weights(model.layers[1].layers[5].get_weights()) # 10%
model_singleoutput2.layers[1].layers[3].set_weights(model.layers[1].layers[8].get_weights()) # 90%


# Extract sample from dataloader
sample = 0
X, y = data_generator[sample]
print(X.shape)

y = np.array(y)
print(y.shape)


# Predict logits for single class >10% contour
single_y_pred, fmaps = model_singleoutput.predict(X)
single_y_pred = single_y_pred[0]

single2_y_pred, fmaps = model_singleoutput2.predict(X)
single2_y_pred = single2_y_pred[0]

single_out = tf.round(tf.nn.sigmoid(single_y_pred))

single2_out = tf.round(tf.nn.sigmoid(single2_y_pred))

roi_diff = tf.math.subtract(single_out, single2_out)

# Create roi 
true_target = np.moveaxis(y[2], 0, -1)


pixel_target = np.zeros_like(true_target)

pixel_target[1000, 1000, 0] = 1


roi = single_out      # All pixles (correctly and incorrectly) predicted as >10% contour
# roi = pixel_target    # A single pixel

# roi = np.ones_like(true_target)  # All pixels


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

print(decoder.summary())
exit()

# Compute gradient w.r.t. bottleneck
bottleneck = tf.Variable(fmaps[0])

with tf.GradientTape() as tape:
    tape.watch(bottleneck)
    decoder_pred = decoder(bottleneck, training = False)[0]
    # y_c = decoder_pred * tf.expand_dims(roi[0], axis=-1)
    y_c = tf.where(roi_diff == 1, decoder_pred, 0)
    y_c = tf.math.reduce_sum(y_c)


# y_c = tf.Variable(y_c)

grads = tape.gradient(y_c, bottleneck)[0]

# print(tf.math.reduce_mean(grads))


# print(grads)

# alpha = np.expand_dims(np.expand_dims(np.mean(grads, axis=(0,1)), 0),0)

# Compute seg-grad-cam
alpha = np.mean(grads, axis=(0,1))

unactivated_cam = np.dot(bottleneck, alpha)

cam = np.maximum(unactivated_cam, 0)

# Normalize
cam = cam / cam.max()

cam = np.moveaxis(cam, 0, -1)

print(cam.shape)

print('resizing cam')
resized_cam = cv2.resize(cam, (0,0), fx = 16, fy=16, interpolation = cv2.INTER_LINEAR)


# plotting
# plt.figure()
# bar = plt.pcolormesh(resized_cam, cmap = 'jet')
# plt.colorbar(bar)
# plt.scatter(1000, 1000, c='white')
# plt.savefig('class_actication_pixel_1000_1000.png')
# plt.savefig('class_actication_map_10%_sample_100.png')
# plt.savefig('resized_class_actication_map_10%_all_pixels_sample_0.png')


fig, ax = plt.subplots(2, 3, figsize=(20,15))

resized_cam = np.ma.masked_where(resized_cam == 0, resized_cam)

norm = mcolors.Normalize(vmin = 0, vmax = 1)

# ax[0, 0].pcolormesh(X[0,...,0], cmap = 'Greys')
ax[0, 0].set_title('Class activation map')
ax[0, 0].pcolormesh(resized_cam, cmap = 'jet', norm = norm)

ax[0, 1].set_title('Recent icechart overlayed')
ax[0, 1].pcolormesh(X[0,...,0], cmap = WMOcolors.cm.sea_ice_chart())
ax[0, 1].pcolormesh(resized_cam, cmap = 'jet', alpha=0.75)

ax[0, 2].set_title('Predicted contour (>=10%)')
ax[0, 2].pcolormesh(single_out[0,...,0], cmap = cmocean.cm.ice)


# ax[0, 2].pcolormesh(X[0,...,1])
# ax[0, 3].pcolormesh(X[0,...,2])

ax[1, 0].set_title('T2M overlayed')
ax[1, 0].pcolormesh(X[0,...,3], cmap = cmocean.cm.thermal)
ax[1, 0].pcolormesh(resized_cam, cmap = 'jet', alpha=0.25)

ax[1, 1].set_title('T2M')
ax[1, 1].pcolormesh(X[0,...,3], cmap = cmocean.cm.thermal)



# ax[1, 1].pcolormesh(X[0,...,4])
# ax[1, 2].pcolormesh(X[0,...,5])

ax[1, 2].set_title('Target contour (>=10%)')
ax[1, 2].pcolormesh(y[5, 0, :, :], cmap = cmocean.cm.ice)

fig.savefig(f'{weights}_difference_Overlay_class10%_sample{sample}_ml_largest2.png')

# gradient_function = K.function([model_singleoutput.input], [bottleneck, grads])
# print((decoder_pred - single_y_pred[0]).mean())
exit()

# single_y_pred = tf.concat(single_y_pred, axis = -1)

single_out = tf.round(tf.nn.sigmoid(single_y_pred))

true_target = y[2, 0]

roi = single_out[0,...,0] * true_target


# compute feature maps gradient



y_c = single_y_pred[0,...,0] * roi

y_c = np.expand_dims(np.expand_dims(y_c, 0), -1)


print(y_c.shape)
print(bottleneck.shape)

grads = tape.gradient(y_c, bottleneck)

print(grads)



# fig, ax = plt.subplots()
# ax.pcolormesh(single_out * true_target)

# plt.savefig('mask_roi.png')