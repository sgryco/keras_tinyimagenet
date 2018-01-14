import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

"""
   This functions comes from 
   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
   I have modified it to work with keras classification outputs.
"""


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
