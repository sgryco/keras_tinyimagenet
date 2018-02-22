"""Definition of the the CNN architecture using the keras API."""

import keras
from keras import regularizers
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling2D, Concatenate, GaussianNoise, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy as top5
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf

"""The metrics used while training the models."""
metrics = ['accuracy', top5, tf.losses.log_loss]


def mirrornet(parameters):
    """The mirrornet model definition.

    The model architecture is dependent on multiple variables:
    parameters.augmentation_strength: influences the Gaussian noise strength
         during training
    parameters.kernel_regularizer: the weights decay
    parameters.nb_filter_inc_per_layer: the number of filters for the first layer
         and the number added for each mirror block
    parameters.conv_repetition: the number of convolution per mirror block
    parameters.filter_size: the size of the convolution filters
    parameters.padding: the type of padding for the convolution

    The training is controlled by
    parameters.optimizer
    parameters.initial_learning_rate

    :returns the keras model.
    """
    regul = regularizers.l2(parameters.kernel_regularizer)
    init = parameters.kernel_initialization

    net_input = Input([64, 64, 3])
    net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(net_input)

    output = net_noise
    local_img = net_noise

    # Define parameters.nb_layers mirror blocks
    for conv_nb, nb_conv in enumerate([parameters.nb_filter_inc_per_layer * i
                                       for i in range(1, parameters.nb_layers + 1)]):
        # Repeat the convolution (feature lane)
        for i in range(parameters.conv_repetition):
            output = Conv2D(nb_conv, kernel_size=parameters.filter_size,
                            strides=1, padding=parameters.padding,
                            kernel_regularizer=regul,
                            kernel_initializer=init,
                            name="block{}_conv{}".format(conv_nb + 1, i + 1)
                            )(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
        output = MaxPooling2D((2, 2))(output)

        # mirror lane: resize the input image to follow the feature lane
        local_img = AveragePooling2D((2, 2), strides=2,
                                     padding=parameters.padding)(local_img)
        output = Concatenate(axis=3)([output, local_img])

    output = Flatten()(output)
    output = Dropout(parameters.dropout_rate)(output)
    output = Dense(200, activation="softmax", kernel_initializer=init, kernel_regularizer=regul)(output)

    model = Model(inputs=net_input, outputs=output)
    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model


def inceptionV3(parameters):
    net_input = Input([64, 64, 3])
    resizer = Lambda(lambda image: keras.backend.resize_images(image, 2.171875, 2.171875, "channels_last"))(net_input)
    # net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(resizer)

    base_model = keras.applications.InceptionV3(
        weights="imagenet" if parameters.pretrained else None,
        include_top=False, input_tensor=resizer)

    output = base_model.output
    output = GlobalAveragePooling2D()(output)
    # output = Dense(1024, activation='relu')(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=base_model.input, outputs=output)
    if not parameters.retrain:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model



def VGG16_custom(parameters):
    """Same VGG-16 architecture as from the keras.application module.

    Weight initialisation can be specified to He_normal to allow training from
    random weights.
    Weights decay can also be adjusted
    """
    regul = regularizers.l2(0.0005)
    init = parameters.kernel_initialization

    img_input = Input([64, 64, 3])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer=init)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer=init)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer=init)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer=init)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer=init)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer=init)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_initializer=init)(x)
    x = Dropout(parameters.dropout_rate)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_initializer=init)(x)
    x = Dropout(parameters.dropout_rate)(x)
    x = Dense(200, activation='softmax', name='predictions', kernel_initializer=init)(x)

    model = Model(inputs=img_input, outputs=x)

    # batch add regularizer
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.add_loss(regul(layer.kernel))

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model




def smallnet(parameters):
    """A previous architecture tested."""
    net_input = Input([64, 64, 3])
    net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(net_input)
    regul = regularizers.l2(parameters.kernel_regularizer)
    init = parameters.kernel_initialization

    output = net_noise
    local_img = net_noise
    for nb_conv in [parameters.nb_filter_inc_per_layer * i for i in range(1, parameters.nb_layers + 1)]:
        for i in range(parameters.conv_repetition):
            output = Conv2D(nb_conv, kernel_size=parameters.filter_size,
                            strides=1, padding=parameters.padding,
                            kernel_regularizer=regul,
                            kernel_initializer=init)(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
        output = MaxPooling2D((2, 2))(output)
        # output = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.9)(output)
        local_img = AveragePooling2D((2, 2), strides=2,
                                     padding=parameters.padding)(local_img)
        output = Concatenate(axis=3)([output, local_img])

    # second
    # merged = AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid")(merged)

    output = Flatten()(output)
    # output = Dense(2600, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    # output = Dropout(parameters.dropout_rate)(output)
    output = Dense(2600, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    output = Dropout(parameters.dropout_rate)(output)
    output = Dense(200, activation="softmax", kernel_initializer=init, kernel_regularizer=regul)(output)

    model = Model(inputs=net_input, outputs=output)
    # should give .47 accuracy

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model


def demo_model(parameters):
    """A model used for demonstration and tests"""
    net_input = Input([64, 64, 3])

    output = Conv2D(32, 5, padding='same', activation='relu')(net_input)
    output = BatchNormalization(epsilon=1e-5, momentum=0.1)(output)
    output = MaxPooling2D((2, 2))(output)

    output = Conv2D(32, 5, padding='same', activation='relu')(output)
    output = BatchNormalization(epsilon=1e-5, momentum=0.1)(output)
    output = MaxPooling2D((2, 2))(output)

    output = Flatten(name='flatten')(output)
    output = Dense(1024, activation="relu")(output)
    output = BatchNormalization(epsilon=1e-5, momentum=0.1)(output)
    output = Dropout(.5)(output)
    output = Dense(200, activation='softmax')(output)

    model = Model(inputs=net_input, outputs=output)

    model.compile(optimizer=SGD(lr=parameters.initial_learning_rate, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model
