import _pickle as pickle
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from .UnPackFile import loadC10

data = loadC10('data_batch_1')

Xtr_rows, Ytr = data
Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]
Yval = Ytr[:1000]

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print('accuracy: %f' % (acc,))

    validation_accuracies.append((k, acc))
