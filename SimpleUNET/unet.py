import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from typing import List


# This initial version is attempting to recreate the architecture of (RONNEBERGER,2015), obviously with the spatial dimensionality of the icechart data.

class convolutional_block(keras.Model):
    def __init__(self, out_channel, kernel_initializer, name='unet_conv_block'):
        super(convolutional_block, self).__init__(name=name)

        self.conv1 = keras.layers.Conv2D(filters = out_channel, kernel_size = 3, padding='same', kernel_initializer=kernel_initializer)
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(filters = out_channel, kernel_size = 3, padding='same', kernel_initializer = kernel_initializer)
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, input_tensor, training = False):
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x, training=training)

        return x

class Encoder(keras.Model):
    def __init__(self, channels, kernel_initializer, name='encoder'):
        super(Encoder, self).__init__(name=name)

        self.encoder_blocks = [convolutional_block(channels[i], kernel_initializer) for i in range(len(channels))]
        self.pool = keras.layers.MaxPool2D()

    def call(self, x, training = False):
        feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            feature_maps.append(x)
            x = self.pool(x)
        
        return feature_maps[::-1]

class Decoder(keras.Model):
    def __init__(self, channels, kernel_initializer, name='decoder'):
        super(Decoder, self).__init__(name=name)

        self.channels = channels
        self.Tconvs = [keras.layers.Conv2DTranspose(filters = channels[i], kernel_size = 2, strides = 2, kernel_initializer=kernel_initializer) for i in range(len(channels))]
        self.decoder_blocks = [convolutional_block(channels[i], kernel_initializer) for i in range(len(channels))]

    def call(self, x, encoder_features, training = False):
        for i in range(len(self.channels)):
            x = self.Tconvs[i](x)
            # cropped_encoder_features = keras.layers.CenterCrop(x.shape[1], x.shape[2])(encoder_features[i])
            x = tf.concat([encoder_features[i], x], axis=-1)
            x = self.decoder_blocks[i](x)

        return x


class UNET(keras.Model):
    def __init__(self, channels, num_classes = 7, kernel_initializer = 'HeNormal', name='unet'):
        super(UNET, self).__init__(name='name')

        self.normalizer = keras.layers.Normalization(axis=-1)
        self.encoder = Encoder(channels = channels, kernel_initializer=kernel_initializer)
        self.decoder = Decoder(channels = channels[:-1][::-1], kernel_initializer=kernel_initializer)
        self.output_layer = keras.layers.Conv2D(filters = num_classes, kernel_size = 1, kernel_initializer = kernel_initializer, dtype=tf.float32)

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training = False):
        x = self.normalizer(x)
        encoder_feature_maps = self.encoder(x)
        x = encoder_feature_maps[0]
        x = self.decoder(x, encoder_feature_maps[1:])
        x = self.output_layer(x)

        x = keras.activations.softmax(x)

        return x


def create_UNET(input_shape: List[int] = (2370, 1844, 6), channels: List[int] = [64, 128, 256], num_classes: int = 7, kernel_initializer: str = 'HeNormal'):
    input = keras.Input(shape=input_shape)
    output = UNET(channels = channels, num_classes = num_classes, kernel_initializer = kernel_initializer)(input)

    model = keras.models.Model(inputs=input, outputs=output)

    return model
