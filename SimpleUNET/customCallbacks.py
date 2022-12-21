import sys
sys.path.append("/mnt/verification_metrics")

import tensorflow as tf
import numpy as np

from tensorflow import keras
from TwoDayForecast.predict_validation import numpy_where_wrapper
from verification_metrics import IIEE
from datetime import datetime, timedelta


class MemoryPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        tf.print(f"\n GPU memory details [current: {float(gpu_dict['current']) / (1024 ** 3)} gb, peak: {float(gpu_dict['peak']) / (1024 ** 3)} gb", output_stream=sys.stdout)


class IIEECallback(keras.callbacks.Callback):
    """Class structure inspired by (https://stackoverflow.com/questions/60080646/access-deprecated-attribute-validation-data-in-tf-keras-callbacks-callback, accessed: 03/12-22)

    Args:
        keras (_type_): _description_
    """

    def __init__(self, validation_data, lsmask, batch_size, ice_edge):
        super(IIEECallback, self).__init__()
        self.validation_data = validation_data
        self.lsmask = lsmask
        self.batches = batch_size * len(self.validation_data)
        self.ice_edge = ice_edge

    def on_epoch_end(self, epoch, logs=None):
        norm_iiee = []

        for batch in range(self.batches):
            yyyymmdd = self.validation_data.get_dates(batch)[0][-13:-5]
            yyyymmdd = datetime.strptime(yyyymmdd, '%Y%m%d')
            mmdd = (yyyymmdd + timedelta(days = 2)).strftime('%m-%d')

            X, y = self.validation_data[batch]
            y = np.concatenate(y, axis = 0)
            y = np.moveaxis(y, 0, -1)
            y = np.apply_along_axis(numpy_where_wrapper, -1, y)[...,0]

            y_pred = self.model.predict(X, verbose = 0)
            y_pred = tf.concat(y_pred, axis = -1)
            y_pred = tf.round(tf.nn.sigmoid(y_pred[0]))
            y_pred = np.apply_along_axis(numpy_where_wrapper, -1, y_pred)[...,0]

            iiee = IIEE(y_pred, y, self.lsmask)
            norm_iiee.append((iiee[0].sum() + iiee[1].sum()) / self.ice_edge[mmdd])

        logs['val_norm_iiee'] = np.mean(norm_iiee)

