import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, Y):
        '''X is N xD where each row is an example. Y is l-dimesion of size N'''
        # the nearest neighbor classifier simply remebers all the training data
        self.Xtr = X
        self.ytr = Y

    def predict(self, X):
        '''X is N x D where each row is  an example we wish to predict label for'''
        num_test = X.shape[0]
        #lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        #loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis =1) #I guess the outcome matrix must have m of Xtr elements.
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest examp
        return Ypred