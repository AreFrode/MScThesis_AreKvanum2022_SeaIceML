import glob
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET
from dataset import HDF5Generator



def main():
    # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)



    SEED_VALUE = 0
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/"
    
    BATCH_SIZE = 2

    data_2019 = np.array(sorted(glob.glob(f"{PATH_DATA}2019/**/*.hdf5", recursive=True)))
    data_2020 = np.array(sorted(glob.glob(f"{PATH_DATA}2020/**/*.hdf5", recursive=True)))
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5", recursive=True)))

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    if not os.path.exists(f"{PATH_OUTPUT}models"):
        os.makedirs(f"{PATH_OUTPUT}models")

    train_generator = HDF5Generator(np.concatenate((data_2019, data_2020)), batch_size=BATCH_SIZE)
    test_generator = HDF5Generator(data_2021, batch_size=BATCH_SIZE)
    
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = 1000000,
        decay_rate = .96,
        staircase=True
    )

    model = create_UNET(input_shape = (960, 896, 6), channels = [64, 128, 256])
    optimizer = keras.optimizers.Adam(learning_rate = 0.01)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = keras.losses.CategoricalCrossentropy(from_logits = True)

    """
    @tf.function
    def train_step(X, y):
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            loss = loss_function(y, predictions)
            scaled_loss = optimizer.get_scaled_loss(loss)
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return predictions, loss

    @tf.function
    def test_step(x):
        print("testing")
        return model(x, training = False)
    """
        
    model.compile(optimizer = optimizer, loss = loss_function, metrics=['accuracy'])

    model.summary()
    # history = model.fit(train_generator, validation_data = validation_generator, epochs = 1, batch_size = BATCH_SIZE)
    
    log_dir = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/logs/fit/{datetime.now().strftime('%d%m%H%M')}"

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    history = model.fit(train_generator, epochs = 20, batch_size = BATCH_SIZE, callbacks=[tensorboard_callback])

    """
    for epoch in range(20):
        print(f"{epoch=}")
        epoch_loss_avg = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy(name = "train_accuracy")
        test_accuracy = keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

        for idx, data in enumerate(train_generator):
            X = data[0]
            y = data[1]
            train_predictions, loss = train_step(X, y)
            epoch_loss_avg.update_state(loss)
            train_accuracy.update_state(y, train_predictions)
            print(f'{idx} / {len(train_generator)}: loss={epoch_loss_avg.result()}, test_accuracy={train_accuracy.result()}')

        for idx, data in enumerate(test_generator):
            print(f"{idx} / {len(test_generator)}")
            X = data[0]
            y = data[1]
            predictions = test_step(X)
            test_accuracy.update_state(y, predictions)

        print(f'Epoch {epoch}: loss={epoch_loss_avg.result()}, train_accuracy={test_accuracy.result()}')
    """

    current_time = datetime.now().strftime("%d%m%H%M")

    print(f"Current time when saving model {current_time}")
    model.save_weights(f'{PATH_OUTPUT}models/weights_{current_time}', save_format='tf')
    # model.save_weights(f'{PATH_OUTPUT}test_model', save_format='tf')
    

if __name__ == "__main__":
    main()