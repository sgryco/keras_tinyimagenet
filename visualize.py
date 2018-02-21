"""
   This functions comes from 
   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
   I have modified it to work with keras classification outputs.
"""
import argparse
import itertools
import keras, os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from callbacks import get_model_weights_file
import keras.preprocessing.image as image
from dataset_loading import mean, std
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(y_test, y_pred, classes_file="class_to_name.txt",
                          normalize=True,
                          title='Normalised Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_test = y_test.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
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

    plt.imshow(cm2)
    plt.colorbar()
    plt.show()

    cm = cm[:10, :10]
    classes = classes[:10]
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

    pass
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                    help='Name this run, required.')
    args = parser.parse_args()
    model_path = get_model_weights_file(args.name)
    print("loading model {}".format(model_path))
    model = keras.models.load_model(model_path, custom_objects={'log_loss': tf.losses.log_loss})
    test_path = "data/val/"
    imgs_path = []
    val_classes = []
    for clid, cl in enumerate(sorted(os.listdir(test_path))):
        cl_path = os.path.join(test_path, cl)
        for im in os.listdir(cl_path):
            imgs_path.append(os.path.join(cl_path, im))
            val_classes.append(clid)
    assert len(imgs_path) == 10000
    imgs_path=imgs_path[:100]
    val_classes=val_classes[:100]
    imgs = np.empty((len(imgs_path), 64, 64, 3), keras.backend.floatx())

    print("loading test images")
    for i, path in enumerate(imgs_path):
        img = image.load_img(path)
        # imgs[i] = preprocess_tf(image.img_to_array(img))
        imgs[i] = (image.img_to_array(img) - mean) / std

    print("predicting")
    predictions = model.predict(imgs)
    a = np.array(val_classes)
    y_true = np.zeros((a.shape[0], 200))
    y_true[np.arange(a.shape[0]), a] = 1
    y_pred = np.zeros((predictions.shape[0], 200))
    y_pred[np.arange(y_pred.shape[0]), np.argmax(predictions, axis=1)] = 1.
    accuracy = accuracy_score(y_true, y_pred=y_pred)
    print("accuracy={}".format(accuracy))

    confusion_matrix(y_true, y_pred)

