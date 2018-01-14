# this file is created by Corentin Cheron chronc@tcd.ie
# some loading functions are originally from https://gitlab.scss.tcd.ie/cs7gv1/tflearn-demo

import sys, os
import h5py


def load_tiny_image_net():
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

