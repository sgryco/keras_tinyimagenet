import os

import keras
import tensorflow as tf
import pygame
import numpy as np
from keras import backend as K

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
        # optimizer = self.model.optimizer
        # lr = K.eval(optimizer.lr * (
        #             1. / (1. + optimizer.decay * optimizer.iterations)))
        # print('\nLR: {:.6f}\n'.format(lr))



def get_model_weights_file(name):
    return os.path.join("checkpoints", name + "-weights.hdf5")


class ManualLR(keras.callbacks.Callback):
     def __init__(self):
        super(ManualLR, self).__init__()
        pygame.init()
        pygame.display.set_mode((100, 100))

     def on_batch_end(self, batch, logs={}):
        update = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.dict['key'] == pygame.K_u:
                update = True
        pygame.event.pump()
        if not update:
            return
        plr = keras.backend.get_value(self.model.optimizer.lr)
        lr = None
        while lr is None:
            val = input("\nCurrent lr: {:.8f}\nInput new learning rate:".format(plr))
            try:
                lr = np.float32(val)
            except ValueError:
                pass
        keras.backend.set_value(self.model.optimizer.lr, lr)
        print('Batch {:05d}: ManualLR new input: {:.8f}'.format(batch + 1, lr))


def get_callbacks(parameters, embedding_layer_names=None):
    tensorboard_dir = os.path.join('.', 'tensorboard', parameters.run_name)

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
    cb_checkpoint = keras.callbacks.ModelCheckpoint(get_model_weights_file(parameters.run_name),
                                                    monitor='val_acc',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)

    cb_tensorboard = LearningRateTracker(parameters=parameters, log_dir=tensorboard_dir, histogram_freq=0,
                                         batch_size=parameters.batch_size,
                                         write_graph=True, write_grads=False,
                                         write_images=True, embeddings_freq=0,
                                         embeddings_layer_names=embedding_layer_names,
                                         embeddings_metadata=None)
    # manual_lr = ManualLR()

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    callbacks = [cb_checkpoint, cb_tensorboard, cb_reduce_lr, cb_early_stop]
    return callbacks
