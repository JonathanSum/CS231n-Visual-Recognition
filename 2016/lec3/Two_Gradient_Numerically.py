import numpy as np


def eval_numerical_gradient(f, x):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)
        x[ix] = old_value

        # partial derivative
        grad[ix] = (fxh - f(x)) / h
        it.iternext()

    return grad


# verson two

# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train
def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)


W = np.random.rand(10, 3073) * 0.001
df = eval_numerical_gradient(CIFAR10_loss_fun(W), W)  # get the gradient

loss_original = CIFAR10_loss_fun(W)
print('original loss %f '(loss_original, ))
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_site_log
    W_new = W - step_size * df
    loss_new=CIFAR10_loss_fun(W_new)

































