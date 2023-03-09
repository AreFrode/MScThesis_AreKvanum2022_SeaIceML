import tensorflow as tf
from tensorflow import keras

from typing import List

def weight_standardization(kernel):
    # See paper on microbatch training https://arxiv.org/abs/1903.10520
    kernel_mean = tf.math.reduce_mean(kernel, axis = [0, 1, 2], keepdims = True, name = 'kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = keras.backend.std(kernel, axis = [0, 1, 2], keepdims = True)
    kernel = kernel / (kernel_std + 1e-5)


class convolutional_block(keras.layers.Layer):
    def __init__(self, out_channel, kernel_initializer, leaky_relu, name='unet_conv_block'):
        super(convolutional_block, self).__init__(name=name)

        self.conv1 = keras.layers.Conv2D(filters = out_channel, kernel_size = 3, padding='same', kernel_initializer=kernel_initializer,)
        # self.bn1 = keras.layers.BatchNormalization()
        
        self.conv2 = keras.layers.Conv2D(filters = out_channel, kernel_size = 3, padding='same', kernel_initializer = kernel_initializer)
        # self.bn2 = keras.layers.BatchNormalization()
        if out_channel < 32:
            self.gn1 = keras.layers.GroupNormalization(groups = int(0.5*out_channel))
            self.gn2 = keras.layers.GroupNormalization(groups = int(0.5*out_channel))
        
        else:
            self.gn1 = keras.layers.GroupNormalization()
            self.gn2 = keras.layers.GroupNormalization()

        if leaky_relu:
            self.activation_function = keras.layers.LeakyReLU(alpha = 0.01)

        else:
            self.activation_function = keras.layers.ReLU()
            

    def call(self, input_tensor, training = False):
        x = self.conv1(input_tensor)
        x = self.activation_function(x)
        # x = keras.activations.linear(x)
        # x = self.bn1(x, training=training)
        x = self.gn1(x)

        x = self.conv2(x)
        x = self.activation_function(x)
        # x = keras.activations.linear(x)
        # x = self.bn2(x, training=training)
        x = self.gn2(x)

        return x

class Encoder(keras.layers.Layer):
    def __init__(self, channels, pooling_factor, average_pool, kernel_initializer, leaky_relu, name='encoder'):
        super(Encoder, self).__init__(name=name)

        self.encoder_blocks = [convolutional_block(channels[i], kernel_initializer, leaky_relu) for i in range(len(channels))]

        if average_pool:
            self.pool = keras.layers.AveragePooling2D(pool_size = pooling_factor)

        else:
            self.pool = keras.layers.MaxPool2D(pool_size = pooling_factor)
            

    def call(self, x, training = False):
        feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            feature_maps.append(x)
            x = self.pool(x)
        
        return feature_maps[::-1]

class ResidualEncoder(keras.layers.Layer):
    def __init__(self, channels, pooling_factor, average_pool, kernel_initializer, leaky_relu, name='encoder'):
        super(ResidualEncoder, self).__init__(name=name)

        self.encoder_blocks = [convolutional_block(channels[i], kernel_initializer, leaky_relu) for i in range(len(channels))]
        self.identity_reshape = [keras.layers.Conv2D(filters=channels[i], kernel_size = 1, kernel_initializer = kernel_initializer) for i in range(len(channels))]

        if average_pool:
            self.pool = keras.layers.AveragePooling2D(pool_size = pooling_factor)

        else:
            self.pool = keras.layers.MaxPool2D(pool_size = pooling_factor)
            

    def call(self, x, training = False):
        feature_maps = []
        for i, block in enumerate(self.encoder_blocks):
            out = block(x)
            identity = self.identity_reshape[i](x)
            x = out + identity
            feature_maps.append(x)
            x = self.pool(x)
        
        return feature_maps[::-1]

class Decoder(keras.layers.Layer):
    def __init__(self, channels, pooling_factor, kernel_initializer, leaky_relu, name='decoder'):
        super(Decoder, self).__init__(name=name)

        self.channels = channels
        self.Tconvs = [keras.layers.Conv2DTranspose(filters = channels[i], kernel_size = pooling_factor, strides = pooling_factor, kernel_initializer=kernel_initializer) for i in range(len(channels))]
        self.decoder_blocks = [convolutional_block(channels[i], kernel_initializer, leaky_relu) for i in range(len(channels))]

    def call(self, x, encoder_features, training = False):
        for i in range(len(self.channels)):
            x = self.Tconvs[i](x)
            # cropped_encoder_features = keras.layers.CenterCrop(x.shape[1], x.shape[2])(encoder_features[i])
            x = tf.concat([encoder_features[i], x], axis=-1)
            x = self.decoder_blocks[i](x)

        return x

class ResidualDecoder(keras.layers.Layer):
    def __init__(self, channels, pooling_factor, kernel_initializer, leaky_relu, name='decoder'):
        super(ResidualDecoder, self).__init__(name=name)

        self.channels = channels
        self.Tconvs = [keras.layers.Conv2DTranspose(filters = channels[i], kernel_size = pooling_factor, strides = pooling_factor, kernel_initializer=kernel_initializer) for i in range(len(channels))]
        self.decoder_blocks = [convolutional_block(channels[i], kernel_initializer, leaky_relu) for i in range(len(channels))]
        self.identity_reshape = [keras.layers.Conv2D(filters=channels[i], kernel_size = 1, kernel_initializer = kernel_initializer) for i in range(len(channels))]

    def call(self, x, encoder_features, training = False):
        for i in range(len(self.channels)):
            x = self.Tconvs[i](x)
            # cropped_encoder_features = keras.layers.CenterCrop(x.shape[1], x.shape[2])(encoder_features[i])
            x = tf.concat([encoder_features[i], x], axis=-1)
            out = self.decoder_blocks[i](x)
            identity = self.identity_reshape[i](x)
            x = out + identity

        return x


class UNET(keras.Model):
    def __init__(self, channels, num_classes = 7, pooling_factor = 2, num_outputs = 7, average_pool = False, kernel_initializer = 'HeNormal', leaky_relu = False, name='unet'):
        super(UNET, self).__init__(name=name)
        # self.normalizer = keras.layers.Normalization(axis=-1)
        self.encoder = Encoder(channels = channels, pooling_factor=pooling_factor, average_pool=average_pool, kernel_initializer=kernel_initializer, leaky_relu=leaky_relu)
        self.decoder = Decoder(channels = channels[:-1][::-1], pooling_factor = pooling_factor, kernel_initializer=kernel_initializer, leaky_relu=leaky_relu)
        self.output_layer = keras.layers.Conv2D(filters = num_classes, kernel_size = 1, kernel_initializer = kernel_initializer, dtype=tf.float32)
        self.num_outputs = num_outputs

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training = False):
        # x = self.normalizer(x)
        encoder_feature_maps = self.encoder(x)
        x = encoder_feature_maps[0]
        x = self.decoder(x, encoder_feature_maps[1:])
        x = self.output_layer(x)

        x = keras.activations.softmax(x)

        return x

class MultiOutputUNET(UNET):
    def __init__(self, channels, num_classes = 1, pooling_factor = 2, num_outputs = 7, average_pool = False, kernel_initializer = 'HeNormal', leaky_relu = False, name = 'unet'):
        UNET.__init__(self, channels, num_classes, pooling_factor, num_outputs, average_pool, kernel_initializer, leaky_relu, name)
        self.output_layers = [keras.layers.Conv2D(filters = num_classes, kernel_size = 1, kernel_initializer = kernel_initializer, dtype=tf.float32, name = f'out{i}') for i in range(self.num_outputs)]
        

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training = False):
        encoder_feature_maps = self.encoder(x)
        x = encoder_feature_maps[0]
        x = self.decoder(x, encoder_feature_maps[1:])

        return [self.output_layers[i](x) for i in range(self.num_outputs)]




def create_UNET(input_shape: List[int] = (2370, 1844, 6), channels: List[int] = [64, 128, 256], num_classes: int = 7, kernel_initializer: str = 'HeNormal'):
    input = keras.Input(shape=input_shape)
    output = UNET(channels = channels, num_classes = num_classes, kernel_initializer = kernel_initializer)(input)

    model = keras.models.Model(inputs=input, outputs=output)

    return model

def create_MultiOutputUNET(input_shape: List[int] = (2370, 1844, 6), channels: List[int] = [64, 128, 256], num_classes: int = 1, pooling_factor = 2, num_outputs = 7, average_pool = False, kernel_initializer: str = 'HeNormal', leaky_relu = False):
    input = keras.Input(shape=input_shape)
    outputs = MultiOutputUNET(channels = channels, num_classes = num_classes, pooling_factor=pooling_factor, num_outputs=num_outputs, average_pool=average_pool, kernel_initializer = kernel_initializer, leaky_relu = leaky_relu)(input)

    model = keras.models.Model(inputs=input, outputs=outputs)

    return model

def main():
    # model = create_MultiOutputUNET((1920, 1840, 9), [64, 128, 256, 512, 1024])
    model = create_MultiOutputUNET((1920, 1840, 9), [64, 128, 256, 512], pooling_factor=2)
    model.summary(expand_nested=True)
    # keras.utils.plot_model(model)


if __name__ == '__main__':
    main()
