"""
This file contains callbacks.

Called between epochs they modify parameters or
log them.

"""

import datetime
import glob
import os
import sys
import time
from shutil import copy

import keras
import tensorflow as tf
import pygame
import numpy as np


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


def get_model_weights_file(name):
    return os.path.join("checkpoints", name + "-weights.hdf5")


class ManualLR(keras.callbacks.Callback):
    """
    Allow to change learning rate at any time during training.

    Just select the small window created by pygame and press 'u'
    You can then enter the new learning rate in the console.

    This callback is disabled for running on AWS, uncomment the two lines below.
    """
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

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    callbacks = [cb_tensorboard, cb_reduce_lr, cb_early_stop]
    # manual_lr = ManualLR()
    # callbacks.append(manual_lr)
    # callbacks.append(cb_checkpoint)
    return callbacks


def save_files(parameters, model):
    """
    Save training parameters for each run.

    All the python files of the project are copied
    A complete description of the network is saved in the file parameter.txt
    All this in the folder saved_parameters.
    """
    root_path = os.path.abspath(os.path.dirname(__file__))
    saved_parameters = os.path.join(root_path, "saved_parameters")
    if not os.path.exists(saved_parameters):
        os.mkdir(saved_parameters)

    # handle timestamp folder
    timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    parameters.timestr = timestr
    folder = os.path.join(saved_parameters, timestr + "_" + parameters.run_name)

    if os.path.exists(folder):
        print("Error {} already exists".format(folder))
        sys.exit(-1)
    os.mkdir(folder)
    for f in glob.glob(os.path.join(root_path, "*.py")):
        copy(f, folder)
    with open(os.path.join(folder, "parameters.txt"), "w") as param_file:
        param_txt = repr(vars(parameters))
        param_file.write(param_txt)
        keras.utils.print_summary(model, print_fn=lambda s: param_file.write(s + '\n'))
        print("Parameters:", param_txt)
    keras.utils.plot_model(model,
                           to_file=os.path.join(folder, timestr + 'model.png'),
                           show_shapes=True,
                           show_layer_names=True)
    from keras.utils.vis_utils import model_to_dot
    dot = model_to_dot(model, show_shapes=False, show_layer_names=True)
    dot.write(os.path.join(folder, timestr + "_graph.dot"))