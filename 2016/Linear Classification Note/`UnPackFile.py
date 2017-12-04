import _pickle as pickle
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import os
from scipy.misc import imread, imsave, imresize



def checking(Xtr):
    for i in range(12):
        print(Xtr[i].shape)


def printSize(data):
    print(len(data[0]))
    print(len(data[0][0]))
    print(len(data[0][0][0]))
    print(len(data[0][0][0][0]))


def loadC10(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')

        X = datadict['data']
        Y = datadict['labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  # 10000,32,32,3
        Y = np.array(Y)
        return X, Y


# data = loadC10('data_batch_1')
# printSize(data)
# print(len(data))
# Xtr, Xte = data
# print(Xte)
# print(Xte.shape)
# img = (Xtr[0])
# plt.subplot(1, 2, 1)
# plt.imshow(img)


# Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
# Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
# dataf= data.reshape(Xtr.shape[0], Xtr.shape[0] * 32 * 3)
