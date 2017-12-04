import DataPack as d
import sys
import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

data = d.loadC10('data_batch_2')
Xtr, Xte = data


def stCorrectIndex(index):
    print("index ",index)
    if index == 0:
        return 1
    if index == 1:
        return 2
    if index == 2:
        return 0


# d.hint()
def answer(arr):
    num = np.argmax(arr, axis=0)
    if num == 0:
        return "0 Bird {} ".format(num)
    elif num == 1:
        return "1 Frog {} ".format(num)
    elif num == 2:
        return "2 Car {} ".format(num)
    else:
        return "Error, element is not in the answer list"


def scoref(index):
    x = Xtr[index].reshape(3072, 1)
    b = -2 + 4 * np.random.rand(10, 1)
    W = -2 + 4 * np.random.rand(10, 3072)
    xForWout = np.append(x, [[1]], axis=0)
    wOut = np.c_[W, b]
    score = np.dot(wOut, xForWout)
    return score


# Frog Car Bird
Xtt = np.append([Xtr[0], Xtr[4]], [Xtr[6]], axis=0)
print("Here is the Xtt " + str(Xtt.shape))

weightG = 0


def st():
    index = 3
    scoreList = np.array([])
    for i in range(index):
        x = Xtt[i].reshape(3072, 1)

        b = -2 + 4 * np.random.rand(3, 1)
        W = -2 + 4 * np.random.rand(3, 3072)
        xForWout = np.append(x, [[1]], axis=0)
        wOut = np.c_[W, b]
        score = np.dot(wOut, xForWout)
        print(answer(score))
        print(score)
        print(score.shape)
    print(scoreList)


def setW():
    W = -2 + 4 * np.random.rand(3, 3072)* 0.001
    return W


# Loss Function, but these for st(simple_test purpose)
def L_i_vst(y, scores):
    delta = 1.0
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

# I don't really use L_ist since vector is the BEST :D I love vector more than numbers
def L_ist(y, W):
    delta = 1.0
    scores = scoref(3)
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes
    loss_i = 0.0
    for j in range(D):
        if j == y:
            continue
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def stw(weight):
    index = 3
    scoreList = np.array([])
    for i in range(index):
        x = Xtt[i].reshape(3072, 1)

        b = -2 + 4 * np.random.rand(3, 1)

        xForWout = np.append(x, [[1]], axis=0)
        wOut = np.c_[weight, b]
        score = np.dot(wOut, xForWout)
        print(answer(score))
        # correcting loss with the vector loss function
        loss = L_i_vst(stCorrectIndex(i), score)
        print("StC is ", stCorrectIndex(i))
        print("Here is our score \n{0} and our loss: {3} \n{1}\n{2}\nour index is {4}\n".format(answer(score), score, score.shape, loss, i))

    print("END")

W1 = setW()
stw(W1)


# def L_i_vst(x, y, W):
#     delta = 1.0
#     scores = score(3)
#     margins = np.maximum(0, scores - scores[y] + delta)
#     margins[y] = 0
#     loss_i = np.sum(margins)
#     return loss_i


def SoftMax(scoreList, correctIndex):
    scoreList = np.array([scoreList[0], scoreList[1], scoreList[2]])

    # Since dividing both side by a C is fine, use a C. Uses logC = -maxj fj is safer, note SVM_debug123
    scoreList -= np.max(scoreList)

    p = np.exp(scoreList) / np.sum(np.exp(scoreList))
    loss_i = (-np.log10(p))[correctIndex]
    return loss_i


np.append([[[1], [2], [3]], [[1], [2], [3]]], [[[3], [5], [7]]], axis=0)
# print(softMax)
print(Xte[2])
print(Xtr[0].shape)

# random number from -2 to 2
# sys.exit()
print("hello")

print("Answer is Frog Car Bird, which is 1,2 and 0 ")


# Gental_Loss Function
def L_i(x, y, W):
    delta = 1.0
    scores = scoref(3)
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes
    loss_i = 0.0
    for j in range(D):
        if j == y:
            continue
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def L_i_vectorized(x, y, W):
    delta = 1.0
    scores = scoref(3)
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
