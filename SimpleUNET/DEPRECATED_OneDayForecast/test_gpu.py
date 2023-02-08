import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
details = tf.config.experimental.get_device_details(gpu_devices[0])
print(details)
