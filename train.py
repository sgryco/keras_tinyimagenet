"""
Main class for running deep net training

"""
import argparse
import datetime
import glob
import os
import sys
import time
from shutil import copy

from callbacks import get_callbacks
from dataset_loading import get_caffe_image_generators
from model import pre_trained_InceptionV3


class Parameters():
    batch_size = 128
    nb_epochs = 80

    initial_learning_rate = 0.01
    lr_patience = 3
    lr_update = .01
    min_lr = .00001
    patience_stop = 5


def main(args):
    # data loading and augmentation
    # train_generator, test_generator = get_normalized_image_generators(Parameters)
    train_generator, test_generator = get_caffe_image_generators(Parameters)

    # define model
    # model = test_model(learning_rate=Parameters.initial_learning_rate)
    model = pre_trained_InceptionV3(learning_rate=Parameters.initial_learning_rate)
    model.summary()
    embedding_layer_names = set(layer.name for layer in model.layers if layer.name.startswith('conv2d_'))

    # TODO:
    # *save model file + parameters to tensorboard folder
    # *require argument model name
    callbacks = get_callbacks(Parameters, embedding_layer_names, model_name=args.name)
    model.fit_generator(train_generator, epochs=Parameters.nb_epochs, verbose=1, validation_data=test_generator,
                        callbacks=callbacks, shuffle='batch', workers=6)
    # fine tune model
    # fine_tune_InceptionV3(model, train_generator, test_generator, callbacks=callbacks, Parameters=Parameters)

    # model.load_weights("./checkpoints/weights.43-0.42.hdf5")
    pass
    pass
    # plot_confusion_matrix(test_y, y_pred)


def save_files(name):
    root_path = os.path.abspath(os.path.basename(__file__))
    saved_parameters = os.path.join(root_path, "saved_parameters")
    if not os.path.exists(saved_parameters):
        os.mkdir(saved_parameters)

    # handle timestamp folder
    timestr = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    folder = timestr + "_" + name

    if os.path.exists(folder):
        print("Error {} already exists".format(folder))
        sys.exit(-1)
    os.mkdir(folder)
    for f in glob.glob(os.path.join(root_path, "*.py")):
        copy(f, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name the this run, required.')
    parser.add_argument('--load', action='store_true', help='Reload previous weights')
    args = parser.parse_args()
    # save files to folder
    save_files(args.name)
    main(args)
