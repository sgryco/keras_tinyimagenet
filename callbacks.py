import os

import keras
import tensorflow as tf


class LearningRateTracker(keras.callbacks.TensorBoard):
    def __init__(self, parameters, **kwargs):
        super(LearningRateTracker, self).__init__(**kwargs)
        self.parameters = parameters

    def set_model(self, model):
        super(LearningRateTracker, self).set_model(model)
        self.summary_op_begin = tf.summary.scalar("Learning_Rate", self.model.optimizer.lr)

        ph = tf.placeholder(tf.string)
        summary_op = tf.summary.text("parameters", ph)
        summarystr = self.sess.run(summary_op, feed_dict={
            ph: [repr(self.parameters.selectedParameters),
                 repr(vars(self.parameters))]
        })
        self.writer.add_summary(summarystr, global_step=0)

    def on_epoch_begin(self, epoch, logs={}):
        super(LearningRateTracker, self).on_epoch_begin(epoch, logs=logs)
        result = self.sess.run(self.summary_op_begin)
        self.writer.add_summary(result, epoch)


def get_callbacks(parameters, embedding_layer_names=None):
    model_name = parameters.run_name
    tensorboard_dir = os.path.join('.', 'tensorboard', model_name)

    # cb_tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
    #                                              batch_size=Parameters.batch_size,
    #                                              write_graph=True, write_grads=False,
    #                                              write_images=True, embeddings_freq=0,
    #                                              embeddings_layer_names=embedding_layer_names,
    #                                              embeddings_metadata=None)
    cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=parameters.lr_update,
                                                     patience=parameters.lr_patience,
                                                     verbose=1,
                                                     min_lr=parameters.min_lr)
    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0., patience=parameters.patience_stop,
                                                  verbose=0, mode='auto')
    cb_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join("checkpoints",
                                                                 model_name + "-weights.hdf5"),
                                                    monitor='val_acc',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=2)

    cb_tensorboard = LearningRateTracker(parameters=parameters, log_dir=tensorboard_dir, histogram_freq=0,
                                         batch_size=parameters.batch_size,
                                         write_graph=True, write_grads=False,
                                         write_images=True, embeddings_freq=0,
                                         embeddings_layer_names=embedding_layer_names,
                                         embeddings_metadata=None)
    os.mkdir("checkpoints")
    callbacks = [cb_tensorboard, cb_reduce_lr, cb_early_stop, cb_checkpoint]
    return callbacks
