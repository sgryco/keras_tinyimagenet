"""
Main class for running deep net training

"""
import argparse
import datetime
import glob
import os
import random
import sys
import time
from shutil import copy

from keras.optimizers import SGD, Adadelta
from keras.utils import print_summary
import keras

from callbacks import get_callbacks
from dataset_loading import get_normalized_image_generators
from itertools import combinations
# noinspection PyPep8Naming
import model as modelFile



class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)


def sgdMomentum095(lr):
    return SGD(lr=lr, momentum=.95)


def sgdMomentum095nesterov(lr):
    return SGD(lr=lr, momentum=.95, nesterov=True)


def adadeltaDefault(lr):
    return Adadelta(lr=lr)


def prelu25():
    return keras.layers.PReLU(
        alpha_initializer=keras.initializers.Constant(value=0.25))

class BaseParameters():
    batch_size = 128
    nb_epochs = 40

    initial_learning_rate = 0.001
    lr_patience = 3
    lr_update = .1
    min_lr = .00001
    patience_stop = 5
    model_name = 'smallnet'
    augmentation_strength = 1.
    optimizer = sgdMomentum095
    dropout_rate = .25
    filter_size = (3, 3)
    nb_layers = 2
    conv_repetition = 3
    nb_filter_inc_per_layer = 16
    padding = "same"
    activation = "relu"
    kernel_initialization = 'glorot_uniform'
    kernel_regularizer = 0.
    loss_function = "categorical_crossentropy"


class PossibleParameters():
    initial_learning_rate = [.01, .001]
    augmentation_strength = [0., 1., 1.5]
    optimizer = [
        sgdMomentum095,
        sgdMomentum095nesterov,
        adadeltaDefault,
    ]
    batch_size = [64, 128, 256]
    dropout_rate = [0., .25, .5]  # effect of dropout, .25 = drop 25%
    # with dropout try momentum between .95-.99

    filter_size = [(3, 3), (5, 5), (7, 7)]    # filter size, 3x3, 5x5, 7x7
    nb_layers = [2, 4, 6] # depth = nb_layer * 2 + 3
    conv_repetition = [1, 3, 6]
    nb_filter_inc_per_layer = [8, 16, 32, 64]  # number of filters, 24, 48, 96
    # padding = ["valid", "same"]
    activation = ["relu", prelu25(), "tanh"]  # prelu25 as from the paper.
    kernel_initialization = ['glorot_uniform', 'he_uniform']
    kernel_regularizer = [0., 5e-4, 1e-3]
    loss_function = ["categorical_crossentropy", "categorical_hinge"]

    # stride, padding,
    # loss functions softmax,



def train(parameters):

    # reset tf graph
    keras.backend.clear_session()

    # data loading and augmentation
    train_generator, test_generator = get_normalized_image_generators(parameters)
    # train_generator, test_generator = get_caffe_image_generators(Parameters)

    # define model
    # model = test_model(learning_rate=Parameters.initial_learning_rate)
    # model = pre_trained_InceptionV3(learning_rate=Parameters.initial_learning_rate)
    # model = VGG16(learning_rate=Parameters.initial_learning_rate, load_weights=False)
    # model = VGG16_custom(learning_rate=Parameters.initial_learning_rate)
    model = getattr(globals()["modelFile"], parameters.model_name)(parameters)
    # model = mynet1(Parameters.initial_learning_rate)

    # save files to folder
    save_files(parameters, model)
    embedding_layer_names = set(layer.name for layer in model.layers if layer.name.startswith('conv2d_'))

    callbacks = get_callbacks(parameters, embedding_layer_names)
    model.fit_generator(train_generator, epochs=parameters.nb_epochs, verbose=1, validation_data=test_generator,
                        callbacks=callbacks, shuffle='batch', workers=6)
    pass
    pass
    # plot_confusion_matrix(test_y, y_pred)


def save_files(parameters, model):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name this run, required.')
    parser.add_argument('--random', default=False, action='store_true',
                        help="Run N (10) random parameters sets")
    parser.add_argument('--independent', default=False, action='store_true',
                        help="Run all parameters sets, changing each variable one at a time")
    parser.add_argument('--nrand', type=int, default=10, help='Number of run')
    parser.add_argument('--load', action='store_true', help='Reload previous weights')
    args = parser.parse_args()

    if args.random:
        for run_id in range(1, args.nrand + 1):
            parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
            parameters['selectedParameters'] = {}
            for key, values in PossibleParameters.__dict__.items():
                if key.startswith('__'):
                    continue
                # import ipdb; ipdb.set_trace()
                param = random.choice(values)
                parameters[key] = param
                parameters['selectedParameters'][key] = param
            parameters = ObjFromDict(parameters)
            parameters.run_name = "{}_{:03d}".format(args.name, run_id)
            train(parameters)
    elif args.independent:
        nb_runs = sum([len(v) - 1 for k, v in PossibleParameters.__dict__.items() if not k.startswith('__')]) + 1
        print("nb total runs = {}".format(nb_runs))
        run_id = 1

        # default run
        parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}_{:03d}_ref".format(args.name, run_id)
        print(sorted(parameters.selectedParameters.items()))
        train(parameters)
        run_id += 1

        for key, values in PossibleParameters.__dict__.items():
            if key.startswith('__'):
                continue

            for value in values:
                if BaseParameters.__dict__[key] == value:
                    continue

                parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
                parameters['selectedParameters'] = {}
                parameters[key] = value
                parameters['selectedParameters'][key] = value
                parameters = ObjFromDict(parameters)
                parameters.run_name = "{}_{:03d}_{}={}".format(args.name, run_id,
                                                               key, value)
                print(sorted(parameters.selectedParameters.items()))
                train(parameters)
                run_id += 1
    else:
        parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
        parameters = ObjFromDict(parameters)
        train(parameters)


if __name__ == "__main__":
    main()
