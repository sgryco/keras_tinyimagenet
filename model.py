import keras
from keras import Sequential
from keras import regularizers
from keras.initializers import he_uniform, glorot_uniform
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling2D, Concatenate, GaussianNoise, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy as top5
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf

metrics = ['accuracy', top5, tf.losses.log_loss]


def test_model(parameters):

    regul = regularizers.l2(0.0001)
    initializer = "he_uniform"
    net_input = Input([64, 64, 3])
    output = net_input

    output = BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3))(output)
    output = Conv2D(32, 3, kernel_regularizer=regul, kernel_initializer=initializer)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling2D((2, 2))(output)

    output = BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3))(output)
    output = Conv2D(64, 3, kernel_regularizer=regul, kernel_initializer=initializer)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling2D((2, 2))(output)


    output = BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3))(output)
    output = Conv2D(96, 3, kernel_regularizer=regul, kernel_initializer=initializer)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling2D((2, 2))(output)

    output = BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3))(output)
    output = Conv2D(239, 3, kernel_regularizer=regul, kernel_initializer=initializer)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling2D((2, 2))(output)


    output = Flatten()(output)
    output = Dense(768, activation='relu', kernel_regularizer=regul, kernel_initializer=initializer)(output)
    output = Dropout(0.25)(output)
    output = Dense(200, activation="softmax", kernel_initializer=initializer, kernel_regularizer=regul)(output)

    model = Model(inputs=net_input, outputs=output)

    sgd = SGD(lr=parameters.initial_learning_rate, decay=0., momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=metrics
                  )
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

def xception(parameters):
    net_input = Input([64, 64, 3])
    resizer = Lambda(lambda image: keras.backend.resize_images(image, 1.109375, 1.109375, "channels_last"))(net_input)

    base_model = keras.applications.Xception(
        weights="imagenet" if parameters.pretrained else None,
        include_top=False, input_tensor=resizer)

    output = base_model.output
    output = Dropout(parameters.dropout_rate)(output)
    output = GlobalAveragePooling2D()(output)
    output = Dense(1024, activation='relu')(output)
    output = Dropout(parameters.dropout_rate)(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=base_model.input, outputs=output)
    if not parameters.retrain:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer=SGD(lr=parameters.initial_learning_rate, momentum=0.9),
    loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def VGG16(parameters):
    net_input = Input([64, 64, 3])
    # resizer = Lambda(lambda image: keras.backend.resize_images(image, 2.171875, 2.171875, "channels_last"))(net_input)

    regul = regularizers.l2(0.0005)
    base_model = keras.applications.VGG16(
        weights="imagenet" if parameters.pretrained else None,
        include_top=False,
        input_tensor=net_input)
    # try to retrain only the last layer?
    # try to lock the old layers?

    output = base_model.output
    output = Flatten()(output)
    output = Dense(4096, activation='relu', kernel_initializer='he_normal')(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, activation='relu', kernel_initializer='he_normal')(output)
    output = Dropout(0.5)(output)
    output = Dense(200, activation="softmax", kernel_initializer='he_normal')(output)

    model = Model(inputs=base_model.input, outputs=output)

    # batch add regularizer to all weights
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.add_loss(regul(layer.kernel))
    if not parameters.retrain:
        for layer in base_model.layers:
            layer.trainable = False
    # from the article:
    # dropout of 0.5
    # lr = 0.01 -> /10 (3 times) (74 epochs for training)
    # weights: random initialisation with small architecture
    # else, take the weights from the small architecture as input
    # -> after submission they found that initialization of Glorot & Bzngio (glorot_uniform)
    # this is the default for Keras

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def VGG16_custom(parameters):
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


def test_regul(learning_rate):
    net_input = Input([64, 64, 3])
    regul = regularizers.l2(0.005)

    output = net_input
    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = MaxPooling2D((2, 2))(output)
    output = BatchNormalization(epsilon=1e-05, momentum=0.9)(output)

    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = MaxPooling2D((2, 2))(output)
    output = BatchNormalization(epsilon=1e-05, momentum=0.9)(output)

    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = Conv2D(32, 3, kernel_regularizer=regul, activation='relu')(output)
    output = MaxPooling2D((2, 2))(output)
    output = BatchNormalization(epsilon=1e-05, momentum=0.9)(output)

    output = Flatten()(output)
    output = Dense(409, activation='relu')(output)
    output = Dense(409, activation='relu')(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=net_input, outputs=output)

    # batch add regularizer
    # for layer in model.layers:
    #     if hasattr(layer, 'kernel'):
    #         layer.add_loss(regul(layer.kernel))

    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def mynet1(parameters):
    net_input = Input([64, 64, 3])
    net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(net_input)
    # regul = regularizers.l2(0.0005)
    regul = None
    init = glorot_uniform()

    output = net_noise
    local_img = net_noise
    for nb_conv in [48, 96, 128, 256]:
        output = Conv2D(nb_conv, kernel_size=3, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(output)
        output = Conv2D(nb_conv, kernel_size=3, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(output)
        output = MaxPooling2D((2, 2))(output)
        output = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.9)(output)
        local_img = AveragePooling2D((2, 2), strides=None, padding="same")(local_img)
        output = Concatenate(axis=3)([output, local_img])
        output = Conv2D(nb_conv, kernel_size=3, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(output)

    # second
    second = net_noise
    local_img = net_noise
    for nb_conv in [48, 96, 128, 256]:
        second = Conv2D(nb_conv, kernel_size=3, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(second)
        second = Conv2D(nb_conv, kernel_size=1, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(second)
        second = MaxPooling2D((2, 2))(second)
        second = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.9)(second)
        local_img = AveragePooling2D((2, 2), strides=None, padding="same")(local_img)
        second = Concatenate(axis=3)([second, local_img])
        second = Conv2D(nb_conv, kernel_size=1, strides=1, padding="same", kernel_regularizer=regul, activation='relu',
                        kernel_initializer=init)(second)

    merged = Concatenate(axis=3)([output, second])
    merged = Conv2D(1024, 1, kernel_regularizer=regul, activation='relu', kernel_initializer=init)(merged)
    merged = MaxPooling2D((4, 4))(merged)
    # merged = AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid")(merged)

    output = Flatten()(merged)
    output = Dense(1392, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    # output = Dropout(0.75)(output)
    output = Dense(768, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    # output = Dropout(0.85)(output)
    output = Dense(200, activation="softmax", kernel_initializer=init, kernel_regularizer=regul)(output)

    model = Model(inputs=net_input, outputs=output)
    # should give .47 accuracy

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def smallnet(parameters):
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


def mirrornet(parameters):
    net_input = Input([64, 64, 3])
    net_noise = GaussianNoise(stddev=.05 * parameters.augmentation_strength)(net_input)
    regul = regularizers.l2(parameters.kernel_regularizer)
    init = parameters.kernel_initialization
    output = net_noise
    local_img = net_noise
    for conv_nb, nb_conv in enumerate([parameters.nb_filter_inc_per_layer * i
                                       for i in range(1, parameters.nb_layers + 1)]):
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
        local_img = AveragePooling2D((2, 2), strides=2,
                                     padding=parameters.padding)(local_img)
        output = Concatenate(axis=3)([output, local_img])

    output = Flatten()(output)
    output = Dropout(parameters.dropout_rate)(output)
    # output = Dense(2600, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    # output = Dropout(parameters.dropout_rate)(output)
    # output = Dense(2600, activation='relu', kernel_regularizer=regul, kernel_initializer=init)(output)
    output = Dense(200, activation="softmax", kernel_initializer=init, kernel_regularizer=regul)(output)

    model = Model(inputs=net_input, outputs=output)
    # should give .47 accuracy

    model.compile(optimizer=parameters.optimizer(parameters.initial_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def resnet50(parameters):
    net_input = Input([64, 64, 3])
    resizer = Lambda(lambda image: keras.backend.resize_images(image, 3.078125, 3.078125, "channels_last"))(net_input)

    base_model = keras.applications.resnet50.ResNet50(
        weights="imagenet" if parameters.pretrained else None,
        include_top=False, input_tensor=resizer, pooling="avg")

    output = base_model.output
    # output = Flatten()(output)
    # output = Dense(1024, activation='relu')(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=SGD(lr=parameters.initial_learning_rate, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model




def demo_model(parameters):
    import tensorflow as tf
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
