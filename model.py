import keras
from keras import Sequential
from keras import regularizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
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