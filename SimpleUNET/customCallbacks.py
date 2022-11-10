import sys
import tensorflow as tf
from tensorflow import keras

class MemoryPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        tf.print(f"\n GPU memory details [current: {float(gpu_dict['current']) / (1024 ** 3)} gb, peak: {float(gpu_dict['peak']) / (1024 ** 3)} gb", output_stream=sys.stdout)