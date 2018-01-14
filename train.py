import sys
import os
import argparse
import keras

from dataset_loading import load_tiny_image_net
from model import test_model
from keras.preprocessing.image import ImageDataGenerator

class Parameters():
    batch_size = 256
    nb_epochs = 40

    initial_learning_rate = 0.1
    lr_patience = 3
    lr_update = .01
    min_lr = .0001
    patience_stop = 5


def main():
    train_x, train_y, test_x, test_y = load_tiny_image_net()
    model = test_model(learning_rate=Parameters.initial_learning_rate)
    model.summary()

    train_generator = ImageDataGenerator(width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         shear_range=.1,
                                         zoom_range=.1,
                                         fill_mode='nearest')
    # train_generator.fit(train_x) # not needed, data already normalised
    test_generator = ImageDataGenerator()
    train_generator = train_generator.flow(train_x, train_y, batch_size=Parameters.batch_size)
    test_generator = test_generator.flow(test_x, test_y, batch_size=Parameters.batch_size)

    cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard/test_model', histogram_freq=0,
                                                 batch_size=Parameters.batch_size,
                                                 write_graph=True, write_grads=False,
                                                 write_images=True, embeddings_freq=1, embeddings_layer_names=None,
                                                 embeddings_metadata=None)
    cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=Parameters.lr_update,
                                                     patience=Parameters.lr_patience,
                                                     verbose=1,
                                                     min_lr=Parameters.min_lr)
    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0., patience=Parameters.patience_stop,
                                                  verbose=1, mode='auto')
    cb_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join("checkpoints",
                                                                 "weights.{epoch:02d}-{val_acc:.2f}.hdf5"),
                                                    monitor='val_acc',
                                                    verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)

    model.fit_generator(train_generator, epochs=Parameters.nb_epochs,
              verbose=1, validation_data=test_generator, validation_steps=(test_x.shape[0] // Parameters.batch_size),
              callbacks=[cb_tensorboard, cb_reduce_lr, cb_early_stop, cb_reduce_lr, cb_checkpoint],
              shuffle='batch', workers=6)


if __name__ == "__main__":
    main()
