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
# noinspection PyPep8Naming
import model as modelFile


class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)


class BaseParameters():
    batch_size = 128
    nb_epochs = 5

    initial_learning_rate = 0.001
    lr_patience = 3
    lr_update = .1
    min_lr = .00001
    patience_stop = 5
    model_name = 'mynet1'
    augmentation_strength = 1.
    optimizer = lambda lr: SGD(lr=lr, momentum=.9)

def sgdMomentum09(lr):
    return SGD(lr=lr, momentum=.9)

def sgdMomentum09nesterov(lr):
    return SGD(lr=lr, momentum=.9, nesterov=True)

def adadeltaDefault(lr):
    return Adadelta(lr=lr)

class PossibleParameters():
    initial_learning_rate = [.01, .001]
    model_name = ['mynet1']
    augmentation_strength = [0., 1., 1.5]
    optimizer = [
        sgdMomentum09,
        sgdMomentum09nesterov,
        adadeltaDefault,
    ]

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
    model.summary()

    # save files to folder
    save_files(parameters)
    embedding_layer_names = set(layer.name for layer in model.layers if layer.name.startswith('conv2d_'))

    callbacks = get_callbacks(parameters, embedding_layer_names)
    model.fit_generator(train_generator, epochs=parameters.nb_epochs, verbose=1, validation_data=test_generator,
                        callbacks=callbacks, shuffle='batch', workers=6)
    pass
    pass
    # plot_confusion_matrix(test_y, y_pred)


def save_files(parameters):
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
        print("Parameters:", param_txt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name this run, required.')
    parser.add_argument('--random', default=False, action='store_true',
                        help="Run N (10) random parameters sets")
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
