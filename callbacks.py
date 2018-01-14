import os

import keras


def get_callbacks(Parameters, embedding_layer_names=None):
    cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard/test_model2', histogram_freq=0,
                                                 batch_size=Parameters.batch_size,
                                                 write_graph=True, write_grads=False,
                                                 write_images=True, embeddings_freq=0,
                                                 embeddings_layer_names=embedding_layer_names,
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

    callbacks = [cb_tensorboard, cb_reduce_lr, cb_early_stop, cb_checkpoint]
    return callbacks
