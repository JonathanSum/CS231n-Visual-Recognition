import DataPack as d
import numpy as np


def debugScores2():
    print("Scores2 is " + str(Scores2.shape))
    # print("Wout is " + str(Wout.shape))
    # print(W1)
    # print(type(xForWout))
    # print(xForWout)
    # print(xForWout.shape)


data = d.loadC10('data_batch_1')
Xtr, Xte = data
# d.hint()
x = Xtr[0].reshape(3072, 1)

# f(xi, W)=W*xi
# Option1: W matrix-multiplies x and then add basic b values
b1 = -2 + 4 * np.random.rand(10, 1)
W1 = -2 + 4 * np.random.rand(10, 3072)
Scores1 = np.dot(W1, x) + b1

# Option2:
b2 = -2 + 4 * np.random.rand(10, 1)
W2 = -2 + 4 * np.random.rand(10, 3072)
xForWout = np.append(x, [[1]], axis=0)
Wout = np.c_[W2, b2]
Scores2 = np.dot(Wout, xForWout)

#option2 is better and faster.
# BigN Notation: if option2 is elements square, then option1 is elements square +b

