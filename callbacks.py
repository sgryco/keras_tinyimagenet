import os

import keras
import tensorflow as tf


class LearningRateTracker(keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(LearningRateTracker, self).__init__(**kwargs)

    def set_model(self, model):
        super(LearningRateTracker, self).set_model(model)
        self.summary_op_begin = tf.summary.scalar("Learning_Rate", self.model.optimizer.lr)
        # self.merged = tf.summary.merge_all()

    def on_epoch_begin(self, epoch, logs={}):
        super(LearningRateTracker, self).on_epoch_begin(epoch, logs=logs)
        result = self.sess.run(self.summary_op_begin)
        self.writer.add_summary(result, epoch)


def get_callbacks(Parameters, embedding_layer_names=None, model_name="default"):
    tensorboard_dir = os.path.join('.', 'tensorboard', model_name)

    # cb_tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
    #                                              batch_size=Parameters.batch_size,
    #                                              write_graph=True, write_grads=False,
    #                                              write_images=True, embeddings_freq=0,
    #                                              embeddings_layer_names=embedding_layer_names,
    #                                              embeddings_metadata=None)
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

    cb_tensorboard = LearningRateTracker(log_dir=tensorboard_dir, histogram_freq=0,
                                         batch_size=Parameters.batch_size,
                                         write_graph=True, write_grads=False,
                                         write_images=True, embeddings_freq=0,
                                         embeddings_layer_names=embedding_layer_names,
                                         embeddings_metadata=None)

    callbacks = [cb_tensorboard, cb_reduce_lr, cb_early_stop, cb_checkpoint]
    return callbacks
