import numpy as np


def l_i_vectorized(x, y, W):
    scores = W.dot(x) #taking a dot product with vector W and x
    margins = np.maximum(0, scores - scores[y] + 1)  #the vector Wx subtracts the score with correct class index, plus one and remove negative number
    margins[y] = 0 #setting the element which the index is the correct class in the loss vector to be zero. Doing that because we don't want to sum up the correct score again.
    loss_i = np.sum(margins) # summing up all the margins in the loss vector
    return loss_i
