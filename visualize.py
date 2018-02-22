# coding=utf-8
"""
   This functions comes from 
   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
   I have modified it to work with keras classification outputs.
"""
import argparse
import itertools

import keras
import keras.preprocessing.image as image
from keras.applications.xception import preprocess_input as preprocess_tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from callbacks import get_model_weights_file
from dataset_loading import mean, std


def plot_confusion_matrix(y_test, y_pred, clid_to_nid,
                          classes_file="class_mapping.txt",
                          normalize=True,
                          title='Normalised Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # y_test = y_test.argmax(axis=1)
    # y_pred = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    with open(classes_file, "r") as cf:
        classes = []
        for line in cf.readlines():
            classes.append(line.split('\t')[1].split(",")[0].strip())

    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # sort classes with regards to accuracy
    accuracies = [(cm[i, i], i) for i in range(cm.shape[0])]
    sorted_acc = sorted(accuracies, key=lambda x: x[0])
    cm2 = np.empty(cm.shape, dtype=cm.dtype)
    for i in range(cm2.shape[0]):
        for o in range(cm2.shape[1]):
            cm2[i, o] = cm[sorted_acc[i][1], sorted_acc[o][1]]
    q1, q3 = np.percentile([a[0] for a in accuracies], [20, 80])
    # import ipdb;ipdb.set_trace()
    # print("q1,q3=", q1, q3)
    # cmcolor = np.empty((cm2.shape[0], cm2.shape[1], 3), np.uint8)
    # cmcolor[cm2 < q1] = 255, 255, 255
    # cmcolor[cm2 >= q1] = 0, 0, 255
    # cmcolor[cm2 >= q3] = 0, 255, 0

    # get worse miss aligned
    cm3 = cm.copy()
    selected = []
    # remove diagonal
    cm3[np.arange(cm3.shape[0]),np.arange(cm3.shape[0])] = 0.

    # import ipdb;ipdb.set_trace()
    while len(selected) < 10:
        maxerr = cm3.max()
        a, b = np.unravel_index(np.argmax(cm3), cm3.shape)
        print("maxerr", maxerr, "at", a, b)
        if a not in selected:
            selected.append(a)
        if b not in selected:
            selected.append(b)
        cm3[a, b] = 0

    selected = list(selected)
    print("selected=", selected)
    cm3 = np.empty((len(selected), len(selected)), np.float32)
    for i in range(cm3.shape[0]):
        for o in range(cm3.shape[1]):
            cm3[i, o] = cm[selected[i], selected[o]]

    # plt.imshow(cmcolor)
    # plt.show()
    #
    #
    # plt.imshow(cm2)
    # plt.colorbar()
    # plt.show()

    # cm = cm2[:10, :10]
    # import ipdb;ipdb.set_trace()
    # cm = cm3
    # classes = [classes[s] for s in selected]
    # # classes = [classes[sorted_acc[i][1]] for i in range(len(cm))]
    # print("selected classes:", classes,
    #       [clid_to_nid[s] for s in selected])
    # print(cm)
    #
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=90)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    #
    # plt.show()

    filename = "confusion_links.csv"
    with open(filename, "w+") as csv:
        csv.write(";".join(['Source', 'Target', 'Weight']))
        csv.write("\n")
        for i in range(cm.shape[0]):
            for n, v in enumerate(cm[i]):
                if v != 0.:
                    csv.write('E{:05};'.format(i))
                    csv.write('E{:05};'.format(n))
                    csv.write("{:.05f}".format(v))
                    csv.write("\n")

    filename = filename.replace(".csv", "_sect.csv")
    print("Exporting to " + filename)
    with open(filename, "w+") as csv:
        csv.write(";".join(['Id', 'Label']))
        csv.write("\n")
        for i in range(cm.shape[0]):
            csv.write('E{:05};'.format(i))
            csv.write(classes[i].split(",")[0])
            csv.write("\n")
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name this run, required.')
    args = parser.parse_args()
    model_path = get_model_weights_file(args.name)
    print("loading model {}".format(model_path))
    model = keras.models.load_model(
        model_path, custom_objects={'log_loss': tf.losses.log_loss})
    test_path = "data/val/"
    imgs_path = []
    val_classes = []
    clid_to_nid = dict()
    for clid, cl in enumerate(sorted(os.listdir(test_path))):
        cl_path = os.path.join(test_path, cl)
        clid_to_nid[clid] = cl
        for im in os.listdir(cl_path):
            imgs_path.append(os.path.join(cl_path, im))
            val_classes.append(clid)
    assert len(imgs_path) == 10000
    imgs_path = imgs_path[:]
    val_classes = val_classes[:]
    imgs = np.empty((len(imgs_path), 64, 64, 3), keras.backend.floatx())

    # import ipdb;ipdb.set_trace()
    print("loading test images")
    for i, path in enumerate(imgs_path):
        img = image.load_img(path)
        imgs[i] = preprocess_tf(image.img_to_array(img))
        # imgs[i] = (image.img_to_array(img) - mean) / std

    print("predicting")
    predictions = model.predict(imgs)

    # transform validation classes into matrix with 1 at the predicted col for each row
    a = np.array(val_classes)
    y_true = np.zeros((a.shape[0], 200))
    y_true[np.arange(a.shape[0]), a] = 1

    # transform prediction into matrix where max = 1 other = 0
    y_pred = np.zeros((predictions.shape[0], 200))
    y_pred[np.arange(y_pred.shape[0]), np.argmax(predictions, axis=1)] = 1.
    accuracy = accuracy_score(y_true, y_pred=y_pred)
    print("accuracy={}".format(accuracy))

    plot_confusion_matrix(val_classes, np.argmax(predictions, axis=1),
                          clid_to_nid)
