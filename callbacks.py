import os

import keras
import tensorflow as tf
from keras import backend as K


class LearningRateTracker(keras.callbacks.Callback):
    def __init__(self, tensorboard_dir):
        super(LearningRateTracker, self).__init__()
        self.writer = tf.summary.FileWriter(tensorboard_dir, filename_suffix="mywriter")
        self.sess = None
        self.summary_op = None

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.summary_op = tf.summary.scalar("Learning_Rate", K.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        result = self.sess.run(self.summary_op)
        self.writer.add_summary(result, epoch)

    def on_train_end(self, _):
        self.writer.close()


def get_callbacks(Parameters, embedding_layer_names=None, model_name="default"):
    tensorboard_dir = os.path.join('.', 'tensorboard', model_name)

    cb_tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
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
                                                                 model_name + "-weights.{epoch:02d}-{val_acc:.2f}.hdf5"),
                                                    monitor='val_acc',
                                                    verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)

    cb_lr_tracker = LearningRateTracker(tensorboard_dir)

    callbacks = [cb_tensorboard, cb_reduce_lr, cb_early_stop, cb_checkpoint, cb_lr_tracker]
    return callbacks
