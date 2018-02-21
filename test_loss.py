import numpy as np

from sklearn.metrics import log_loss
import math

def logloss(true_label, predicted, eps=1e-7):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -math.log(p)
    else:
        return -math.log(1 - p)

nc = 200
nrow = 1
y_true = np.zeros((nrow, nc))
y_true[np.arange(nrow), np.arange(nrow)%nc] = 1.

y_pred = np.ones((nrow, nc)) * 1. / nc

# import ipdb;ipdb.set_trace()
loss = log_loss(y_true, y_pred)


print("sk learn log loss", loss)
closs = 0.
for i in range(y_true.shape[0]):
    for j in range(y_true.shape[1]):
        closs += logloss(y_true[i, j], y_pred[i, j])
closs /= y_true.shape[0] * y_true.shape[1]
print("Closs =", closs)

#
#
#
#
# conclusion,
# tf learn does this for each element of each line:
# -y_true*log(y_pred)-(1-y_true)*log(1-y_pred)/Nclasses

# sklearn does:
# -y_true*log(y_pred)

# then they both average on all the lines (samples)
