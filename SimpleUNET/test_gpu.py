import tensorflow as tf

print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")