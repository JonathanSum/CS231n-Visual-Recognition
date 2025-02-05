import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype= self.ytr.dtype)

        for i in range(num_test):
            #l1
            distances = np.sum(np.abs(self.Xtr-X[i,:]), axis=1)
            #l2
            distances= np.sqrt(self.Xtr-X[i,:], axis=1)
            min_index=np.argmin(distances)
            Ypred[i]=self.ytr[min_index]

        return Ypred
