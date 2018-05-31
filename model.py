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

from keras.applications.imagenet_utils import (
    preprocess_input, _obtain_input_shape, decode_predictions)
from keras.applications.resnet50 import (
    identity_block, conv_block, WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP)
from keras.layers import ZeroPadding2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.utils.data_utils import get_file


def ResNet50_lr(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=64,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model


def resnet50(parameters):
    net_input = Input([64, 64, 3])
    # resizer = Lambda(lambda image: keras.backend.resize_images(image, 2, 2, "channels_last"))(net_input)
    resizer = net_input
    # net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(resizer)

    base_model = ResNet50_lr(
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
