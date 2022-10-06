import glob
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras
from unet import create_UNET
from dataset import HDF5Generator



def main():
    SEED_VALUE = 0
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"
    
    BATCH_SIZE = 1

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

    model = create_UNET(input_shape = (1920, 1840, 6), channels = [64, 128, 256, 512, 1024])
    optimizer = keras.optimizers.Adam(learning_rate = 0.01)
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = keras.losses.CategoricalCrossentropy(from_logits = True)

    @tf.function
    def train_step(X, y):
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            loss = loss_function(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return predictions, loss

    @tf.function
    def test_step(x):
        return model(x, training = False)
        
    # model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss=keras.losses.CategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

    model.summary()
    # history = model.fit(train_generator, validation_data = validation_generator, epochs = 1, batch_size = BATCH_SIZE)
    # history = model.fit(train_generator, epochs = 20, batch_size = BATCH_SIZE)

    for epoch in range(20):
        print(f"{epoch=}")
        epoch_loss_avg = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy(name = "train_accuracy")
        test_accuracy = keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

        for idx, data in enumerate(train_generator):
            print(f"{idx} / {len(train_generator)}", end = '\r')
            X = data[0]
            y = data[1]
            train_predictions, loss = train_step(X, y)
            epoch_loss_avg.update_state(loss)
            train_accuracy.update_state(y, train_predictions)
        print('\n')

        for idx, data in enumerate(test_generator):
            print(f"{idx} / {len(test_generator)}", end = '\r')
            X = data[0]
            y = data[1]
            test_predictions = test_step(X)
            test_accuracy.update_state(y, test_predictions)
        print('\n')

        print(f'Epoch {epoch}: loss={epoch_loss_avg.result()}, train_accuracy={train_accuracy.result()} test_accuracy={test_accuracy.result()}')

    current_time = datetime.now().strftime("%d%m%H%M")

    print(f"Current time when saving model {current_time}")
    model.save_weights(f'{PATH_OUTPUT}models/weights_{current_time}', save_format='tf')
    # model.save_weights(f'{PATH_OUTPUT}test_model', save_format='tf')
    

if __name__ == "__main__":
    main()