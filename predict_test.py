import argparse

import keras, os
import numpy as np

from callbacks import get_model_weights_file
from keras.applications.xception import preprocess_input as preprocess_tf
import keras.preprocessing.image as image
import tensorflow as tf
from model import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true')
parser.add_argument('--name', type=str, required=True,
                    help='Name this run, required.')
args = parser.parse_args()

model_path = get_model_weights_file(args.name)
print("loading model {}".format(model_path))
model = keras.models.load_model(model_path, custom_objects={'log_loss': tf.losses.log_loss})

if args.val:
    test_path = "data/val/"
    imgs_path = []
    val_classes = []
    for clid, cl in enumerate(sorted(os.listdir(test_path))):
        cl_path = os.path.join(test_path, cl)
        for im in os.listdir(cl_path):
            imgs_path.append(os.path.join(cl_path, im))
            val_classes.append(clid)
    assert len(imgs_path) == 10000
    imgs_path=imgs_path
    val_classes=val_classes
else:
    test_path = "test_data/test_data_raw"
    imgs_path = []
    for cl in sorted(os.listdir(test_path)):
        cl_path = os.path.join(test_path, cl)
        imgs_path.append(cl_path)
    assert len(imgs_path) == 10000

imgs = np.empty((len(imgs_path), 64, 64, 3), keras.backend.floatx())

print("loading test images")
for i, path in enumerate(imgs_path):
    img = image.load_img(path)
    imgs[i] = preprocess_tf(image.img_to_array(img))

print("predicting")
predictions = model.predict(imgs)
import math
def logloss(true_label, predicted, eps=1e-7):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -math.log(p)
  else:
    return -math.log(1 - p)

if args.val:
    from sklearn.metrics import log_loss, accuracy_score
    a = np.array(val_classes)
    y_true = np.zeros((a.shape[0], 200))
    y_true[np.arange(a.shape[0]), a] = 1
    # import ipdb;ipdb.set_trace()
    model.compile('sgd', loss='categorical_crossentropy', metrics=metrics)
    ev = model.evaluate(imgs, y_true)
    print(ev)
    loss = log_loss(y_true, predictions)
    closs = 0.
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
           closs += logloss(y_true[i, j], predictions[i, j])
    closs /= y_true.shape[0] * y_true.shape[1]
    print("Closs =", closs)

    y_pred = np.zeros((predictions.shape[0], 200))
    y_pred[np.arange(y_pred.shape[0]), np.argmax(predictions, axis=1)] = 1.
    accuracy = accuracy_score(y_true, y_pred=y_pred)
    print("log loss={}, accuracy={}".format(loss, accuracy))

else:
    with open("kaggle_" + args.name + ".txt", "w") as txtfile:
        txtfile.write("imid,n01443537,n01629819,n01641577,n01644900,n01698640,n01742172,"
                      "n01768244,n01770393,n01774384,n01774750,n01784675,n01855672,"
                      "n01882714,n01910747,n01917289,n01944390,n01945685,n01950731,"
                      "n01983481,n01984695,n02002724,n02056570,n02058221,n02074367,"
                      "n02085620,n02094433,n02099601,n02099712,n02106662,n02113799,"
                      "n02123045,n02123394,n02124075,n02125311,n02129165,n02132136,"
                      "n02165456,n02190166,n02206856,n02226429,n02231487,n02233338,"
                      "n02236044,n02268443,n02279972,n02281406,n02321529,n02364673,"
                      "n02395406,n02403003,n02410509,n02415577,n02423022,n02437312,"
                      "n02480495,n02481823,n02486410,n02504458,n02509815,n02666196,"
                      "n02669723,n02699494,n02730930,n02769748,n02788148,n02791270,"
                      "n02793495,n02795169,n02802426,n02808440,n02814533,n02814860,"
                      "n02815834,n02823428,n02837789,n02841315,n02843684,n02883205,"
                      "n02892201,n02906734,n02909870,n02917067,n02927161,n02948072,"
                      "n02950826,n02963159,n02977058,n02988304,n02999410,n03014705,"
                      "n03026506,n03042490,n03085013,n03089624,n03100240,n03126707,"
                      "n03160309,n03179701,n03201208,n03250847,n03255030,n03355925,"
                      "n03388043,n03393912,n03400231,n03404251,n03424325,n03444034,"
                      "n03447447,n03544143,n03584254,n03599486,n03617480,n03637318,"
                      "n03649909,n03662601,n03670208,n03706229,n03733131,n03763968,"
                      "n03770439,n03796401,n03804744,n03814639,n03837869,n03838899,"
                      "n03854065,n03891332,n03902125,n03930313,n03937543,n03970156,"
                      "n03976657,n03977966,n03980874,n03983396,n03992509,n04008634,"
                      "n04023962,n04067472,n04070727,n04074963,n04099969,n04118538,"
                      "n04133789,n04146614,n04149813,n04179913,n04251144,n04254777,"
                      "n04259630,n04265275,n04275548,n04285008,n04311004,n04328186,"
                      "n04356056,n04366367,n04371430,n04376876,n04398044,n04399382,"
                      "n04417672,n04456115,n04465501,n04486054,n04487081,n04501370,"
                      "n04507155,n04532106,n04532670,n04540053,n04560804,n04562935,"
                      "n04596742,n04597913,n06596364,n07579787,n07583066,n07614500,"
                      "n07615774,n07695742,n07711569,n07715103,n07720875,n07734744,"
                      "n07747607,n07749582,n07753592,n07768694,n07871810,n07873807,"
                      "n07875152,n07920052,n09193705,n09246464,n09256479,n09332890,"
                      "n09428293,n12267677\n")
        for i in range(predictions.shape[0]):
            txtfile.write(
                ",".join([str(i)] + ["{:.05f}".format(p) for p in predictions[i]])+ "\n")
    print("done!")
