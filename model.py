import keras
from keras import Sequential
from keras import regularizers
from keras.layers import GlobalAveragePooling2D, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD


def test_model(learning_rate=0.01,
               loss_function='categorical_crossentropy',
               num_classes=200):

    regul = regularizers.l2(0.0001)
    initializer = "glorot_uniform"
    model = Sequential()

    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3)))
    model.add(Conv2D(32, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(Conv2D(32, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3)))

    model.add(Conv2D(64, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(Conv2D(64, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3)))

    model.add(Conv2D(128, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(Conv2D(128, 3, kernel_regularizer=regul, kernel_initializer=initializer,
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, input_shape=(64, 64, 3)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=regul,
                    kernel_initializer=initializer))
    model.add(Dropout(0.15))  # the rate of dropping, here 15%
    model.add(Dense(1024, activation='relu', kernel_regularizer=regul,
                    kernel_initializer=initializer))
    model.add(Dropout(0.15))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))

    sgd = SGD(lr=learning_rate, decay=0., momentum=0.9, nesterov=True)

    # should get 40% val acc
    top5 = keras.metrics.top_k_categorical_accuracy
    model.compile(loss=loss_function,
                  optimizer=sgd,
                  metrics=['accuracy', top5]
                  )
    return model


def pre_trained_InceptionV3(learning_rate):
    net_input = Input([64, 64, 3])
    resizer = Lambda(lambda image: keras.backend.resize_images(image, 2.171875, 2.171875, "channels_last"))(net_input)

    base_model = keras.applications.InceptionV3(weights=None, include_top=False, input_tensor=resizer)

    output = base_model.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(1024, activation='relu')(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=base_model.input, outputs=output)
    # for layer in base_model.layers:
    #     layer.trainable = False

    top5 = keras.metrics.top_k_categorical_accuracy
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy', top5])

    return model


def fine_tune_InceptionV3(model, train_generator, test_generator, callbacks, Parameters):
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    top5 = keras.metrics.top_k_categorical_accuracy
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy', top5])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator, epochs=Parameters.nb_epochs,
                        verbose=1, validation_data=test_generator,
                        callbacks=callbacks, shuffle='batch', workers=6)


def VGG16(learning_rate):
    net_input = Input([64, 64, 3])
    resizer = Lambda(lambda image: keras.backend.resize_images(image, 2.171875, 2.171875, "channels_last"))(net_input)

    base_model = keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=resizer)

    output = base_model.output
    output = Flatten(output)
    output = Dense(2048, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(2048, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(200, activation="softmax")(output)

    model = Model(inputs=base_model.input, outputs=output)
    # for layer in base_model.layers:
    #     layer.trainable = False
    # from the article:
    # dropout of 0.5
    # lr = 0.01 -> /10 (3 times) (74 epochs for training)
    # weights: random initialisation with small architecture
    # else, take the weights from the small architecture as input
    # -> after submission they found that initialization of Glorot & Bzngio (glorot_uniform)
    # this is the default for Keras

    top5 = keras.metrics.top_k_categorical_accuracy
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top5])

    return model
