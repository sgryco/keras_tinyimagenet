# this file is created by Corentin Cheron chronc@tcd.ie
# some loading functions are originally from https://gitlab.scss.tcd.ie/cs7gv1/tflearn-demo

import os
import sys

import h5py
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


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


def get_normalized_image_generators_from_hdf5(parameters):
    train_x, train_y, test_x, test_y = load_normalised_tiny_image_net_from_h5()
    strength = parameters.augmentation_strength
    train_generator = ImageDataGenerator(width_shift_range=0.15 * strength,
                                         height_shift_range=0.15 * strength,
                                         horizontal_flip=True,
                                         shear_range=.2 * strength,
                                         zoom_range=.15 * strength,
                                         # zca_whitening=True if strength != 0. else False,
                                         # zca_epsilon=1e-6 * strength,
                                         fill_mode='reflect')
    # train_generator.fit(train_x)

    # for X_batch, y_batch in train_generator.flow(train_x, train_y, batch_size=9):
    #     # create a grid of 3x3 images
    #     for i in range(0, 9):
    #         pyplot.subplot(330 + 1 + i)
    #         pyplot.imshow(X_batch[i], cmap=pyplot.get_cmap('gray'))
    #     # show the plot
    #     pyplot.show()
    #     break

    # sys.exit(0)

    train_generator = train_generator.flow(train_x, train_y, batch_size=parameters.batch_size)
    test_generator = ImageDataGenerator()
    test_generator = test_generator.flow(test_x, test_y, batch_size=parameters.batch_size, shuffle=False)

    return train_generator, test_generator


def create_hdf5_from_numpy(imgs, classes, path):
    dataset = h5py.File(path, 'w')
    dataset.create_dataset('X', data=imgs)
    dataset.create_dataset('Y', data=classes)
    dataset.close()


def get_normalized_image_generators(parameters):
    mean, std = None, None
    mean = np.array([[[122.4626756, 114.25840613, 101.37467571]]],
                    dtype=np.float32)
    std = np.array([[[70.63153376, 68.6114437, 71.93088608]]],
                   dtype=np.float32)
    if mean is None or std is None:
        tmp_generator = ImageDataGenerator().flow_from_directory(
            "./data/train",
            shuffle=False,
            batch_size=1000,
            target_size=parameters.input_size)

        print("loading image to compute normalization...")
        sums = np.zeros((3), dtype=np.int64)
        nump = 0
        for i in range(100000//1000):
            print("mean {}/100\r".format(i + 1), end="")
            img, val = next(tmp_generator)
            sums += img.astype(np.int64).sum(axis=(0,1,2))
            nump += img.shape[0] * img.shape[1] * img.shape[2]
        mean = (sums.astype(np.float64) / nump).reshape(1, 1, 3)
        print("mean={}".format(mean))

        # compute std
        sums = np.zeros((3), dtype=np.float64)
        nump = 0
        for i in range(100000//1000):
            print("std {}/100\r".format(i + 1), end="")
            img, val = next(tmp_generator)
            sums += np.square((img.astype(np.float64) - mean)).sum(axis=(0,1,2))
            nump += img.shape[0] * img.shape[1] * img.shape[2]
        std = np.sqrt((sums.astype(np.float64) / nump)).reshape(1, 1, 3)
        print("std={}".format(std))


    strength = parameters.augmentation_strength
    train_data_generator = ImageDataGenerator(featurewise_std_normalization=True,
                                         featurewise_center=True,
                                         horizontal_flip=True,
                                         width_shift_range=0.15 * strength,
                                         height_shift_range=0.15 * strength,
                                         shear_range=.2 * strength,
                                         zoom_range=.15 * strength,
                                         fill_mode='reflect')
    train_data_generator.mean, train_data_generator.std = mean, std
    train_generator = train_data_generator.flow_from_directory(
        "./data/train",
        target_size=parameters.input_size,
        batch_size=parameters.batch_size, shuffle=True)

    # import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    # imgs = np.empty((1,1),dtype=np.float64)
    # imgs, classes = next(train_generator)

    val_generator = ImageDataGenerator(featurewise_std_normalization=True,
                                       featurewise_center=True)
    val_generator.mean, val_generator.std = mean, std
    val_generator = val_generator.flow_from_directory(
        "./data/val",
        target_size=parameters.input_size,
        color_mode="rgb",
        batch_size=parameters.batch_size, shuffle=False)
    return train_generator, val_generator
