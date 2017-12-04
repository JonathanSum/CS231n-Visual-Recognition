import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.ytr = Y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            #l2
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            distances=distances**(0.5)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
