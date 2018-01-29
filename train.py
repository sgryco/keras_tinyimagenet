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

from callbacks import get_callbacks, get_model_weights_file
from dataset_loading import get_normalized_image_generators, get_tf_image_generator
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
    batch_size = 64
    nb_epochs = 71
    input_size = (64, 64)
    pretrained = False

    initial_learning_rate = 0.001
    lr_patience = 2
    lr_update = .1
    min_lr = .000001
    patience_stop = 5
    model_name = 'smallnet'
    augmentation_strength = 1.2
    optimizer = sgdMomentum095
    dropout_rate = 0.12
    filter_size = (5, 5)
    nb_layers = 4
    conv_repetition = 3
    nb_filter_inc_per_layer = 64
    padding = "same"
    activation = "relu"
    kernel_initialization = 'glorot_uniform'
    kernel_regularizer = 0.001
    loss_function = "categorical_crossentropy"
    retrain = False  # by default do not retrain base model (that have pretrained weights)


class PossibleParameters():
    initial_learning_rate = [.01, .001]
    augmentation_strength = [0., 1., 1.5]
    optimizer = [
        sgdMomentum095,
        sgdMomentum095nesterov,
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

sequenceToRun = [
    # {"kernel_initialization": 'glorot_uniform', "dropout_rate": 0.},
    # {"kernel_initialization": 'he_uniform', "dropout_rate": 0.},
    # {"kernel_initialization": 'he_uniform', "dropout_rate": 0.12},
    # {"kernel_initialization": 'glorot_uniform', "dropout_rate": 0.12},
    {"kernel_initialization": 'he_uniform', "dropout_rate": 0.25},
    {"kernel_initialization": 'he_uniform', "dropout_rate": 0.37},
    {"kernel_initialization": 'he_uniform', "dropout_rate": 0.50},
]



def train(parameters):
    # reset tf graph
    keras.backend.clear_session()

    # data loading and augmentation
    # train_generator, test_generator = get_normalized_image_generators(parameters)
    train_generator, test_generator = get_tf_image_generator(parameters)

    # define model
    model = getattr(globals()["modelFile"], parameters.model_name)(parameters)
    if hasattr(parameters, "load") and parameters.load:
        weight_file = get_model_weights_file(parameters.load)
        print("Loading weights from: {}".format(parameters.load))
        model.load_weights(weight_file)

    # save files to folder
    save_files(parameters, model)
    embedding_layer_names = set(layer.name for layer in model.layers if layer.name.startswith('conv2d_'))

    callbacks = get_callbacks(parameters, embedding_layer_names)
    model.fit_generator(train_generator, epochs=parameters.nb_epochs, verbose=1, validation_data=test_generator,
                        callbacks=callbacks, shuffle='batch', workers=8, use_multiprocessing=True)
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
    # keras.utils.plot_model(model,
    #                        to_file=os.path.join(folder, timestr + 'model.png'),
    #                        show_shapes=True,
    #                        show_layer_names=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name this run, required.')
    parser.add_argument('--random', default=False, action='store_true',
                        help="Run N (10) random parameters sets")
    parser.add_argument('--model', help="The name of the model (function name in model.py)")
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--sequence', default=False, action='store_true')
    parser.add_argument('--load', type=str, help='load previous weights')
    parser.add_argument('--independent', default=False, action='store_true',
                        help="Run all parameters sets, changing each variable one at a time")
    parser.add_argument('--nrand', type=int, default=10, help='Number of run')
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

        for key, values in sorted(PossibleParameters.__dict__.items()):
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
    elif args.sequence:
        run_id = 1
        for seq_dict in sequenceToRun:
            parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
            parameters['selectedParameters'] = {}
            for key, value in seq_dict.items():
                if key not in parameters:
                    raise ValueError("key {} not in base parameters".format(key))
                parameters[key] = value
                parameters['selectedParameters'][key] = value
            parameters = ObjFromDict(parameters)
            parameters.run_name = "{}_{:03d}".format(args.name, run_id)
            print(sorted(parameters.selectedParameters.items()))
            train(parameters)
            run_id += 1

    elif args.model:
        parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        for key, val in [("model_name", args.model), ("pretrained", args.pretrained),
                         ("retrain", args.retrain), ("load", args.load)]:
            parameters[key] = val
            parameters['selectedParameters'][key] = val
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}".format(args.name)
        train(parameters)

    else:
        parameters = {k: v for k, v in BaseParameters.__dict__.items() if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}".format(args.name)
        train(parameters)


if __name__ == "__main__":
    main()
