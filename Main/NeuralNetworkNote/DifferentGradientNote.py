import numpy as np


def problem(x):
    e = 2.71828182845904590
    return x ** 3 + 2 * x + e ** x - 3


def error(x):
    pointwewant = 1
    return (problem(x) - pointwewant) ** 2


def derivativef(x):
    vSNum = 0.0001  # Very Small Numer for the delta in the derivative equation
    derivative = (error(x + vSNum) - error(x - vSNum)) / (2 * vSNum)
    return derivative


def derivative_descent(x):
    derivative = derivativef(x)

    middleMan = 0.01  # This man can't be too big or too small because the decent will be either moving slow or overshoot.
    """
    x = x - derivative * middleMan  # I think it is better to write it as -derivative * middleMan+x
    """
    mu, v = 0.5, 0

    # Momentum or I call it friction update
    # x = mu**v-derivative * middleMan

    # NesterovMomentumUpdate
    x_ahead = x + mu * v
    dx_ahead = derivativef(x_ahead)
    v = mu * v - middleMan * dx_ahead
    x += v
    return x


x = 0.0
for i in range(50):
    x = derivative_descent(x)
    print('Step:{}, x ={:6f}, problem(x) = {:6f}, Error = {:6f}'.format(i, x, problem(x), error(x)))

# Adagrad is an adaptive learning rate method originally proposed by Duchi et al..
# Assume the gradient dx and parameter vector x
dx = derivative_descent(x)
cache, eps, learning_rate = 0, 1e-5, 0.5
cache += dx ** 2
x += - learning_rate * dx / (np.sqrt(cache) + eps)


# RMSProp with momentum
def RMSP(x):
    dx = derivative_descent(x)
    cache, eps, learning_rate, decay_rate = 0, 1e-7, 0.5, 0.7
    cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
    x += - learning_rate * dx / (np.sqrt(cache) + eps)


# Adam update

def AdamUpdate(x):
    beta1, beta2,v,m = 0.9, 0.995,0.5,0.5
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    x += -learning_rate * m / (np.sqrt(v) + 1e-7)
