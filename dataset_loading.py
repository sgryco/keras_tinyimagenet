# this file is created by Corentin Cheron chronc@tcd.ie
# some loading functions are originally from https://gitlab.scss.tcd.ie/cs7gv1/tflearn-demo

import os
import sys

import h5py
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def load_normalised_tiny_image_net_from_h5():
    # Check if hdf5 databases already exist and create them if not.
    if (not os.path.exists('hdf5/tiny-imagenet_train.h5') or
            not os.path.exists('hdf5/tiny-imagenet_val.h5')):
        print("Error, hdf5 dataset not found in folder hdf5.")
        sys.exit(-1)

    # Load training data from hdf5 dataset.
    h5f = h5py.File('hdf5/tiny-imagenet_train.h5', 'r')
    train_x = h5f['X'][:]
    train_y = h5f['Y'][:]

    # Load validation data.
    h5f = h5py.File('hdf5/tiny-imagenet_val.h5', 'r')
    test_x = h5f['X'][:]
    test_y = h5f['Y'][:]

    return train_x, train_y, test_x, test_y


def get_normalized_image_generators(Parameters):
    train_x, train_y, test_x, test_y = load_raw_tiny_image_net_from_h5()
    train_generator = ImageDataGenerator(width_shift_range=0.05,
                                         height_shift_range=0.05, horizontal_flip=True,
                                         shear_range=.1, zoom_range=.1, fill_mode='nearest')
    train_generator = train_generator.flow(train_x, train_y, batch_size=Parameters.batch_size)
    test_generator = ImageDataGenerator()
    test_generator = test_generator.flow(test_x, test_y, batch_size=Parameters.batch_size, shuffle=False)

    return train_generator, test_generator


def get_caffe_image_generators(Parameters):
    # more info on: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    # In the paper, the model is denoted as the configuration D trained with scale jittering.
    # The input images should be zero-centered by mean pixel (rather than mean image) subtraction.
    # Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].

    train_x, train_y, test_x, test_y = load_raw_tiny_image_net_from_h5()
    train_generator = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=0.05,
                                         height_shift_range=0.05, horizontal_flip=True,
                                         shear_range=.1, zoom_range=.1, fill_mode='nearest')
    train_generator = train_generator.flow(train_x, train_y, batch_size=Parameters.batch_size)
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_generator.flow(test_x, test_y, batch_size=Parameters.batch_size, shuffle=False)

    return train_generator, test_generator


def load_raw_tiny_image_net_from_h5():
    # Check if hdf5 databases already exist and create them if not.
    if (not os.path.exists('hdf5_raw/tiny-imagenet_train.h5') or
            not os.path.exists('hdf5_raw/tiny-imagenet_val.h5')):
        print("Error, hdf5 raw dataset not found in folder hdf5.")
        sys.exit(-1)

    # Load training data from hdf5 dataset.
    h5f = h5py.File('hdf5_raw/tiny-imagenet_train.h5', 'r')
    train_x = h5f['X'][:]
    train_y = h5f['Y'][:]

    # Load validation data.
    h5f = h5py.File('hdf5_raw/tiny-imagenet_val.h5', 'r')
    test_x = h5f['X'][:]
    test_y = h5f['Y'][:]

    return train_x, train_y, test_x, test_y


def load_raw_tiny_image_net_from_files(preprocess=None, target_size=(224, 224), batch_size=128):
    val_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                       data_format="channels_last")
    with open("./data/tiny-imagenet-200/val/val_annotations.txt", "r") as val_file:
        val_classes = []
        img_paths = []
        for line in val_file.readlines():
            img_path, class_id = line.split()[:2]
            val_classes.append(class_id)
            img_paths.append(img_path)

    val_generator = val_generator.flow_from_directory("./data/tiny-imagenet-200/val", target_size=target_size,
                                                      color_mode="rgb",
                                                      batch_size=batch_size, shuffle=False, follow_links=True)
    return val_generator, val_classes
